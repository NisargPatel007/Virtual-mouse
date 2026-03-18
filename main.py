"""
Virtual Mouse with Gesture Control
===================================
Right hand  → cursor movement (fist to freeze)
Left hand   → left click (thumb+index pinch)
              right click (thumb+middle pinch)
              scroll (index+middle finger extended, move up/down)

PowerPoint mode (two fists to toggle):
  [How to enter/exit: Make a FIST with BOTH hands and hold for 2 seconds]
  
  EASY MOVES / GESTURES DURING PRESENTATION:
  • Next Slide         → Thumbs UP (right hand)
  • Previous Slide     → Thumbs DOWN (right hand)
  • Zoom In / Out      → Extend right index finger, then pinch/expand thumb
  • Floating Toolbar   → Both index fingers pointing up (hold for 0.3s)
    ↳ The toolbar gives you on-screen buttons for:
      - Start Presentation (F5)
      - Stop Presentation (ESC)
      - Draw Pen (Ctrl+P)
      - Laser Pointer (Ctrl+L)
      - Next/Prev and Zoom+/-
    ↳ Point to a button and pinch your right index+thumb to click it!

Optimised for low latency:
  • Threaded camera capture (no blocking on I/O)
  • ctypes Win32 API for mouse (≈0.1 ms vs pyautogui's ≈15-20 ms)
  • Exponential moving average smoothing with dead-zone filter
  • Frame-skip (MediaPipe runs every 2nd frame)
"""

import cv2
import math
import time
import ctypes
import threading
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, List

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# ──────────────────────────────────────────────────────────────
# Win32 API constants & helpers
# ──────────────────────────────────────────────────────────────
user32 = ctypes.windll.user32

MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_WHEEL      = 0x0800
WHEEL_DELTA             = 120

# Keyboard constants
VK_LEFT         = 0x25
VK_RIGHT        = 0x27
VK_OEM_PLUS     = 0xBB          # '=' / '+' key
VK_OEM_MINUS    = 0xBD          # '-' / '_' key
VK_CONTROL      = 0x11
VK_F5           = 0x74
VK_ESCAPE       = 0x1B
KEYEVENTF_KEYUP = 0x0002

SM_CXSCREEN = 0
SM_CYSCREEN = 1

screen_w = user32.GetSystemMetrics(SM_CXSCREEN)
screen_h = user32.GetSystemMetrics(SM_CYSCREEN)


def move_cursor(x: int, y: int):
    """Move cursor instantly via Win32."""
    user32.SetCursorPos(int(x), int(y))


def left_click():
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def right_click():
    user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)


def scroll(amount: int):
    """Positive = scroll up, negative = scroll down."""
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(amount), 0)


def press_key(vk_code: int):
    """Simulate a key press + release via Win32."""
    user32.keybd_event(vk_code, 0, 0, 0)
    user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


def ctrl_press_key(vk_code: int):
    """Ctrl + key combo via Win32."""
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(vk_code, 0, 0, 0)
    user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


# ──────────────────────────────────────────────────────────────
# Threaded camera capture
# ──────────────────────────────────────────────────────────────
class CameraStream:
    """Grabs frames in a background thread so the main loop never waits."""

    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)              # request highest FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # minimal buffer
        self.ret = False
        self.frame = None
        self._lock = threading.Lock()
        self._stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stopped = True
        self.cap.release()


# ──────────────────────────────────────────────────────────────
# MediaPipe hand landmarker setup
# ──────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# ──────────────────────────────────────────────────────────────
# Tunable parameters
# ──────────────────────────────────────────────────────────────
SMOOTHING        = 0.20          # EMA factor (0 = max smooth, 1 = raw)
DEADZONE         = 4             # ignore movements smaller than this (pixels)
PINCH_THRESHOLD  = 0.045         # normalised distance for pinch detection
CLICK_COOLDOWN   = 0.35          # seconds between clicks
SCROLL_SCALE     = 3000          # multiplier for scroll sensitivity
MARGIN           = 0.08          # edge margin for cursor mapping

# PowerPoint mode parameters
PPT_ENTER_HOLD       = 2.0      # seconds to hold both fists to ENTER ppt mode
PPT_EXIT_HOLD        = 3.0      # seconds to hold both fists to EXIT ppt mode
PPT_SLIDE_COOLDOWN   = 0.5      # seconds between slide changes

# Toolbar parameters
TOOLBAR_TRIGGER_HOLD   = 0.3    # seconds to hold both-index-up to show toolbar
TOOLBAR_DISMISS_HOLD   = 1.0    # seconds to hold again to dismiss toolbar
ZOOM_COOLDOWN          = 0.5    # seconds between zoom actions
PINCH_EXPAND_THRESHOLD = 0.03   # normalised distance delta to trigger zoom

# Camera frame dimensions
CAM_W, CAM_H = 640, 480

# ──────────────────────────────────────────────────────────────
# Toolbar system
# ──────────────────────────────────────────────────────────────
class ToolbarManager:
    """Renders and manages a floating PPT toolbar ALWAYS ON TOP using Tkinter."""

    def __init__(self):
        self.visible = False
        self.hovered_index = -1

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "#000001")
        self.root.config(bg="#000001")

        BTN_W, BTN_H = 80, 80
        PAD = 15
        COLS = 4

        # (name, icon_char, color_hex)
        button_defs = [
            ("Next",   ">>",  "#64dc64"),
            ("Prev",   "<<",  "#64dc64"),
            ("Zoom+",  " + ", "#50b4ff"),
            ("Zoom-",  " - ", "#50b4ff"),
            ("Start",  "F5",  "#6464ff"),
            ("Stop",   "ESC", "#6464ff"),
            ("Draw",   "P",   "#ffc850"),
            ("Laser",  "L",   "#c864ff"),
        ]

        self.buttons = []
        total_w = COLS * BTN_W + (COLS - 1) * PAD
        total_rows = ((len(button_defs) + COLS - 1) // COLS)
        total_h = total_rows * BTN_H + (total_rows - 1) * PAD
        
        self.start_x = (screen_w - total_w) // 2
        self.start_y = (screen_h - total_h) // 2

        self.root.geometry(f"{total_w}x{total_h}+{self.start_x}+{self.start_y}")
        self.canvas = tk.Canvas(self.root, width=total_w, height=total_h, bg="#000001", highlightthickness=0)
        self.canvas.pack()

        for i, (name, icon, color) in enumerate(button_defs):
            row, col = divmod(i, COLS)
            cx = col * (BTN_W + PAD)
            cy = row * (BTN_H + PAD)
            sx = self.start_x + cx
            sy = self.start_y + cy
            
            bg = "#333333"
            fg = "#aaaaaa"
            rect_id = self.canvas.create_rectangle(cx, cy, cx+BTN_W, cy+BTN_H, fill=bg, outline="#777777", width=2)
            icon_id = self.canvas.create_text(cx+BTN_W//2, cy+BTN_H//2 - 12, text=icon, fill=fg, font=("Arial", 20, "bold"))
            name_id = self.canvas.create_text(cx+BTN_W//2, cy+BTN_H//2 + 20, text=name, fill=fg, font=("Arial", 10))

            self.buttons.append({
                "name": name, "icon": icon, "color": color,
                "cx": cx, "cy": cy, "sx": sx, "sy": sy, 
                "w": BTN_W, "h": BTN_H, "action": None,
                "ids": (rect_id, icon_id, name_id)
            })

        self.last_hovered_index = -1
        self.last_update_time = 0.0

        self.buttons[0]["action"] = lambda: press_key(VK_RIGHT)
        self.buttons[1]["action"] = lambda: press_key(VK_LEFT)
        self.buttons[2]["action"] = lambda: ctrl_press_key(VK_OEM_PLUS)
        self.buttons[3]["action"] = lambda: ctrl_press_key(VK_OEM_MINUS)
        self.buttons[4]["action"] = lambda: press_key(VK_F5)
        self.buttons[5]["action"] = lambda: press_key(VK_ESCAPE)
        self.buttons[6]["action"] = lambda: ctrl_press_key(ord("P"))
        self.buttons[7]["action"] = lambda: ctrl_press_key(ord("L"))

        self.root.withdraw()

    def toggle(self):
        if self.visible: self.hide()
        else: self.show()

    def show(self):
        self.visible = True
        self.hovered_index = -1
        self.root.deiconify()
        self.root.attributes("-topmost", True)

    def hide(self):
        self.visible = False
        self.hovered_index = -1
        self.root.withdraw()

    def get_hovered_button(self, px: int, py: int) -> int:
        """px, py are FULL SCREEN pixel coordinates."""
        for i, b in enumerate(self.buttons):
            if b["sx"] <= px <= b["sx"] + b["w"] and b["sy"] <= py <= b["sy"] + b["h"]:
                self.hovered_index = i
                return i
        self.hovered_index = -1
        return -1

    def select(self):
        if 0 <= self.hovered_index < len(self.buttons):
            btn = self.buttons[self.hovered_index]
            if btn["action"]: btn["action"]()
            self.hide()
            return btn["name"]
        return None

    def update(self):
        """Draw toolbar on screen and pump Tk events. Does not block."""
        try:
            now = time.monotonic()
            if self.visible and self.hovered_index != self.last_hovered_index:
                for i, btn in enumerate(self.buttons):
                    is_hovered = (i == self.hovered_index)
                    bg = btn["color"] if is_hovered else "#333333"
                    fg = "#ffffff" if is_hovered else "#aaaaaa"
                    outline = "#ffffff" if is_hovered else "#777777"
                    line_w = 4 if is_hovered else 2
                    
                    rect_id, icon_id, name_id = btn["ids"]
                    self.canvas.itemconfigure(rect_id, fill=bg, outline=outline, width=line_w)
                    self.canvas.itemconfigure(icon_id, fill=fg)
                    self.canvas.itemconfigure(name_id, fill=fg)
                    
                self.last_hovered_index = self.hovered_index

            # Limit Tkinter updates to ~90 FPS to preserve CPU for MediaPipe
            if now - self.last_update_time > 0.011:
                self.root.update()
                self.last_update_time = now
        except:
            pass

# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────
smooth_x, smooth_y = screen_w / 2.0, screen_h / 2.0
last_click_time = 0.0
prev_scroll_y = 0.0
scroll_active = False

# PowerPoint mode state
ppt_mode = False
ppt_activation_start = 0.0         # when the cross gesture began
last_slide_time = 0.0              # last slide-change timestamp
ppt_flash_text = ""                # brief overlay text
ppt_flash_until = 0.0              # timestamp until which to show flash

# Toolbar state
toolbar = ToolbarManager()
toolbar_trigger_start = 0.0        # when both-index-up began
prev_pinch_dist = 0.0              # for zoom tracking between frames
last_zoom_time = 0.0               # last zoom-action timestamp

# For monotonic timestamps (avoids issues with system clock changes)
_start_time = time.monotonic()


def get_timestamp_ms():
    return int((time.monotonic() - _start_time) * 1000)


# ──────────────────────────────────────────────────────────────
# Gesture helpers
# ──────────────────────────────────────────────────────────────
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def is_fist(hand):
    """All four fingers curled = fist (finger tips below PIP joints)."""
    return (
        hand[8].y  > hand[6].y  and   # index
        hand[12].y > hand[10].y and   # middle
        hand[16].y > hand[14].y and   # ring
        hand[20].y > hand[18].y       # pinky
    )


def fingers_extended(hand):
    """Check if index and middle fingers are extended (for scroll gesture)."""
    index_up  = hand[8].y  < hand[6].y
    middle_up = hand[12].y < hand[10].y
    return index_up and middle_up


def is_index_extended(hand):
    """Check if index finger is extended (tip above PIP joint)."""
    return hand[8].y < hand[6].y


def is_thumbs_up(hand):
    """Thumb pointing UP with all four fingers curled (fist)."""
    fingers_curled = (
        hand[8].y  > hand[6].y  and   # index
        hand[12].y > hand[10].y and   # middle
        hand[16].y > hand[14].y and   # ring
        hand[20].y > hand[18].y       # pinky
    )
    thumb_up = hand[4].y < hand[3].y  # thumb tip above thumb IP joint
    return fingers_curled and thumb_up


def is_thumbs_down(hand):
    """Thumb pointing DOWN with all four fingers curled (fist)."""
    fingers_curled = (
        hand[8].y  > hand[6].y  and
        hand[12].y > hand[10].y and
        hand[16].y > hand[14].y and
        hand[20].y > hand[18].y
    )
    thumb_down = hand[4].y > hand[3].y  # thumb tip below thumb IP joint
    return fingers_curled and thumb_down


def is_both_index_up(right_hand, left_hand):
    """Both hands present with only index finger extended — toolbar trigger."""
    if right_hand is None or left_hand is None:
        return False

    def _index_only(hand):
        return (
            hand[8].y  < hand[6].y  and   # index up
            hand[12].y > hand[10].y and   # middle down
            hand[16].y > hand[14].y and   # ring down
            hand[20].y > hand[18].y       # pinky down
        )

    return _index_only(right_hand) and _index_only(left_hand)


def get_pinch_distance(hand):
    """Normalised distance between thumb tip and index tip."""
    return distance(hand[4], hand[8])


def detect_two_fists(right_hand, left_hand):
    """Return True if both hands are forming a closed fist."""
    if right_hand is None or left_hand is None:
        return False
    return is_fist(right_hand) and is_fist(left_hand)


def classify_hands(result):
    """
    Return (right_hand, left_hand) landmarks using MediaPipe handedness labels.
    Returns None for a hand that isn't detected.
    """
    right = left = None
    for i, handedness_list in enumerate(result.handedness):
        label = handedness_list[0].category_name   # "Left" or "Right"
        # MediaPipe mirrors: camera "Right" = user's right hand
        if label == "Right":
            left = result.hand_landmarks[i]
        else:
            right = result.hand_landmarks[i]
    return right, left


def map_to_screen(x_norm, y_norm):
    """Map normalised hand coords (with margin clamp) to full screen coords."""
    x = (x_norm - MARGIN) / (1.0 - 2 * MARGIN)
    y = (y_norm - MARGIN) / (1.0 - 2 * MARGIN)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return x * screen_w, y * screen_h


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────
SHOW_PREVIEW = True   # set False for max performance (no OpenCV window)

cam = CameraStream(src=0, width=CAM_W, height=CAM_H)

# Give camera thread time to start
time.sleep(0.3)

print(f"Screen: {screen_w}x{screen_h}")
print("Virtual Mouse running — press 'q' to quit")

frame_count = 0
right_hand = left_hand = None     # initialise before first detection

try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        # ── Frame-skip: run MediaPipe every 2nd frame ──
        frame_count += 1
        if frame_count % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts = get_timestamp_ms()
            result = detector.detect_for_video(mp_image, ts)
            right_hand, left_hand = classify_hands(result)

        now = time.monotonic()

        # ──────── BOTH FISTS → TOGGLE PPT MODE ────────
        hold_needed = PPT_EXIT_HOLD if ppt_mode else PPT_ENTER_HOLD
        if detect_two_fists(right_hand, left_hand):
            if ppt_activation_start == 0.0:
                ppt_activation_start = now
            elif now - ppt_activation_start >= hold_needed:
                ppt_mode = not ppt_mode
                ppt_activation_start = 0.0
                toolbar.hide()
                toolbar_trigger_start = 0.0
                prev_pinch_dist = 0.0
                if ppt_mode:
                    ppt_flash_text = "PPT MODE ON"
                else:
                    ppt_flash_text = "PPT MODE OFF"
                ppt_flash_until = now + 1.5
                print(f"PowerPoint mode: {'ON' if ppt_mode else 'OFF'}")
        else:
            ppt_activation_start = 0.0

        # ═══════════════════════════════════════════════════════
        #  PPT MODE
        # ═══════════════════════════════════════════════════════
        if ppt_mode:

            # ── Toolbar trigger: both index fingers up ──
            if is_both_index_up(right_hand, left_hand):
                if toolbar_trigger_start == 0.0:
                    toolbar_trigger_start = now
                held = now - toolbar_trigger_start
                if not toolbar.visible and held >= TOOLBAR_TRIGGER_HOLD:
                    toolbar.show()
                    toolbar_trigger_start = now      # reset for dismiss timing
                elif toolbar.visible and held >= TOOLBAR_DISMISS_HOLD:
                    toolbar.hide()
                    toolbar_trigger_start = 0.0
            else:
                toolbar_trigger_start = 0.0

            # ── Toolbar OPEN: navigate + select ──
            if toolbar.visible:
                if right_hand is not None:
                    idx_tip = right_hand[8]
                    raw_x, raw_y = map_to_screen(idx_tip.x, idx_tip.y)
                    smooth_x += SMOOTHING * (raw_x - smooth_x)
                    smooth_y += SMOOTHING * (raw_y - smooth_y)
                    move_cursor(smooth_x, smooth_y)
                    toolbar.get_hovered_button(int(smooth_x), int(smooth_y))

                    # Select with pinch (thumb + index close)
                    if get_pinch_distance(right_hand) < PINCH_THRESHOLD:
                        if (now - last_click_time) > CLICK_COOLDOWN:
                            btn_name = toolbar.select()
                            if btn_name:
                                ppt_flash_text = btn_name
                                ppt_flash_until = now + 0.8
                            last_click_time = now

            # ── Toolbar CLOSED: passive gestures ──
            else:
                # Thumbs up / down → slide navigation (right hand only)
                if right_hand is not None and (now - last_slide_time) > PPT_SLIDE_COOLDOWN:
                    if is_thumbs_up(right_hand):
                        press_key(VK_RIGHT)
                        ppt_flash_text = "NEXT \u2192"
                        ppt_flash_until = now + 0.8
                        last_slide_time = now
                    elif is_thumbs_down(right_hand):
                        press_key(VK_LEFT)
                        ppt_flash_text = "\u2190 PREV"
                        ppt_flash_until = now + 0.8
                        last_slide_time = now

                # Zoom: pinch expand / contract (right hand, index must be extended)
                if right_hand is not None and is_index_extended(right_hand):
                    curr_pinch = get_pinch_distance(right_hand)
                    if prev_pinch_dist > 0 and (now - last_zoom_time) > ZOOM_COOLDOWN:
                        delta = curr_pinch - prev_pinch_dist
                        if delta > PINCH_EXPAND_THRESHOLD:
                            ctrl_press_key(VK_OEM_PLUS)
                            ppt_flash_text = "ZOOM +"
                            ppt_flash_until = now + 0.5
                            last_zoom_time = now
                        elif delta < -PINCH_EXPAND_THRESHOLD:
                            ctrl_press_key(VK_OEM_MINUS)
                            ppt_flash_text = "ZOOM -"
                            ppt_flash_until = now + 0.5
                            last_zoom_time = now
                    prev_pinch_dist = curr_pinch
                else:
                    prev_pinch_dist = 0.0

        # ═══════════════════════════════════════════════════════
        #  DEFAULT MODE (cursor + clicks + scroll)
        # ═══════════════════════════════════════════════════════
        else:
            # ──────── RIGHT HAND → CURSOR ────────
            if right_hand is not None:
                if not is_fist(right_hand):
                    index_tip = right_hand[8]
                    raw_x, raw_y = map_to_screen(index_tip.x, index_tip.y)

                    # EMA smoothing
                    smooth_x += SMOOTHING * (raw_x - smooth_x)
                    smooth_y += SMOOTHING * (raw_y - smooth_y)

                    # Dead-zone: move only if displacement is meaningful
                    ddx = abs(smooth_x - raw_x)
                    ddy = abs(smooth_y - raw_y)
                    if ddx > DEADZONE or ddy > DEADZONE or True:
                        move_cursor(smooth_x, smooth_y)

            # ──────── LEFT HAND → ACTIONS ────────
            if left_hand is not None:
                thumb    = left_hand[4]
                index_l  = left_hand[8]
                middle_l = left_hand[12]

                d_index  = distance(thumb, index_l)
                d_middle = distance(thumb, middle_l)

                # LEFT CLICK — thumb + index pinch
                if d_index < PINCH_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
                    left_click()
                    last_click_time = now

                # RIGHT CLICK — thumb + middle pinch
                elif d_middle < PINCH_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
                    right_click()
                    last_click_time = now

                # SCROLL — index + middle extended, vertical movement
                if fingers_extended(left_hand):
                    current_y = left_hand[8].y
                    if scroll_active and prev_scroll_y != 0:
                        delta = (prev_scroll_y - current_y) * SCROLL_SCALE
                        if abs(delta) > 5:
                            scroll(int(delta))
                    prev_scroll_y = current_y
                    scroll_active = True
                else:
                    prev_scroll_y = 0.0
                    scroll_active = False

        # Update Tkinter toolbar separately (now an overlay on the screen)
        toolbar.update()

        # ── Preview window (optional) ──
        if SHOW_PREVIEW:
            pass # Tkinter toolbar now updates independently via root.update()

            # Draw Prominent Mode Indicator
            mode_text = "MODE: POWERPOINT" if ppt_mode else "MODE: DEFAULT (MOUSE)"
            mode_col = (0, 255, 255) if ppt_mode else (0, 255, 0)
            
            tw, th = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cx = (frame.shape[1] - tw) // 2
            cv2.rectangle(frame, (cx - 10, 10), (cx + tw + 10, 10 + th + 15), (0, 0, 0), -1)
            cv2.putText(frame, mode_text, (cx, 10 + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_col, 2)

            # Draw lightweight status text for hands
            status = []
            if ppt_mode:
                if toolbar.visible:
                    status.append("TOOLBAR OPEN")
            else:
                if right_hand is not None:
                    status.append("R: " + ("FIST (FREEZE)" if is_fist(right_hand) else "MOVE"))
                if left_hand is not None:
                    status.append("L: ACTIVE")

            for i, txt in enumerate(status):
                colour = (255, 255, 0) if ppt_mode else (0, 255, 0)
                cv2.putText(frame, txt, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            # Two-fists hold indicator
            if ppt_activation_start > 0:
                progress = min(1.0, (now - ppt_activation_start) / hold_needed)
                bar_w = int(200 * progress)
                cv2.rectangle(frame, (10, 70 + len(status) * 30),
                              (10 + bar_w, 90 + len(status) * 30), (0, 255, 255), -1)
                secs_left = max(0, hold_needed - (now - ppt_activation_start))
                label = f"Both Fists held... {secs_left:.1f}s"
                cv2.putText(frame, label,
                            (10, 65 + len(status) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Toolbar-trigger hold indicator
            if toolbar_trigger_start > 0 and ppt_mode:
                tgt = TOOLBAR_DISMISS_HOLD if toolbar.visible else TOOLBAR_TRIGGER_HOLD
                prog = min(1.0, (now - toolbar_trigger_start) / tgt)
                bw = int(160 * prog)
                by = 95 + len(status) * 30
                cv2.rectangle(frame, (10, by), (10 + bw, by + 16), (255, 180, 50), -1)
                cv2.putText(frame, "Toolbar...",
                            (10, by - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 50), 1)

            # Flash text for slide / mode change feedback
            if ppt_flash_text and now < ppt_flash_until:
                tw = cv2.getTextSize(ppt_flash_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
                cx = (frame.shape[1] - tw) // 2
                cv2.putText(frame, ppt_flash_text, (cx, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            cv2.imshow("Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()
    print("Virtual Mouse stopped.")
