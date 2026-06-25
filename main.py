"""
Virtual Mouse with Gesture Control  (v4)
=========================================

DEFAULT MODE
────────────
  Right hand  → cursor movement   (fist = freeze cursor)
  Left hand   → left click        (thumb + index pinch)
                right click       (thumb + middle pinch)
                double click      (thumb + index + middle all pinched)
                scroll            (index + middle extended, move up/down)

ENTER / EXIT POWERPOINT MODE
  Both fists held 2s → toggle ON / OFF

POWERPOINT MODE  — right hand only
──────────────────────────────────
  Cursor FROZEN. Left hand ignored.

  ┌───────────────────────────────────────────────────────────┐
  │ 1.  Both fists 2s          → Exit PPT mode               │
  │ 2.  Both index up 0.3s     → Show / hide toolbar         │
  │ 3.  Three fingers hold 1s  → Start slideshow (F5)        │
  │ 4.  Fist hold 1.5s         → Stop slideshow (Esc)        │
  │ 5.  Index only ☝️           → Next slide                  │
  │ 6.  Peace ✌️                → Previous slide              │
  │ 7.  Open palm 🖐️ + move    → Zoom in / out               │
  │ 8.  Fist hold 0.8s         → Freeze / unfreeze cursor    │
  └───────────────────────────────────────────────────────────┘
"""

import cv2
import math
import time
import ctypes
import threading
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# ─────────────────────────────────────────────────────────────
# Win32 helpers
# ─────────────────────────────────────────────────────────────
user32 = ctypes.windll.user32

MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP   = 0x0010
MOUSEEVENTF_WHEEL     = 0x0800
WHEEL_DELTA           = 120
KEYEVENTF_KEYUP       = 0x0002

VK_LEFT      = 0x25
VK_RIGHT     = 0x27
VK_CONTROL   = 0x11
VK_F5        = 0x74
VK_ESCAPE    = 0x1B
VK_L         = 0x4C
VK_OEM_PLUS  = 0xBB
VK_OEM_MINUS = 0xBD

screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)


def move_cursor(x: int, y: int):
    user32.SetCursorPos(int(x), int(y))

def _mouse(flag, data=0):
    user32.mouse_event(flag, 0, 0, data, 0)

def left_click():
    _mouse(MOUSEEVENTF_LEFTDOWN); _mouse(MOUSEEVENTF_LEFTUP)

def right_click():
    _mouse(MOUSEEVENTF_RIGHTDOWN); _mouse(MOUSEEVENTF_RIGHTUP)

def double_click():
    left_click(); time.sleep(0.04); left_click()

def scroll(amount: int):
    _mouse(MOUSEEVENTF_WHEEL, int(amount))

def press_key(vk: int):
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

def ctrl_press_key(vk: int):
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


# ─────────────────────────────────────────────────────────────
# Threaded camera
# ─────────────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self.ret = False; self.frame = None
        self._lock = threading.Lock(); self._stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret; self.frame = frame

    def read(self):
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self._stopped = True; self.cap.release()


# ─────────────────────────────────────────────────────────────
# MediaPipe
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────
SMOOTHING        = 0.20
DEADZONE         = 4
PINCH_THRESH     = 0.045
SCROLL_SCALE     = 2500
MARGIN           = 0.08

LCLICK_COOLDOWN  = 0.30
RCLICK_COOLDOWN  = 0.30
DCLICK_COOLDOWN  = 0.40

PPT_TOGGLE_HOLD       = 2.0    # both fists held this long → toggle mode
PPT_SLIDE_COOLDOWN    = 0.6
THREE_FINGER_HOLD     = 1.0    # hold 3-finger this long → start slideshow
FIST_STOP_HOLD        = 1.5    # hold fist this long → stop slideshow

TOOLBAR_TRIGGER_HOLD  = 0.3
TOOLBAR_DISMISS_HOLD  = 1.0

ZOOM_VERT_DEADZONE    = 0.018
ZOOM_REPEAT_DELAY     = 0.35

SLIDE_CONFIRM_FRAMES  = 3      # frames to confirm index/peace for slide

CURSOR_FREEZE_HOLD    = 0.8    # hold fist 0.8s → toggle cursor freeze

CAM_W, CAM_H = 640, 480


# ─────────────────────────────────────────────────────────────
# Toolbar
# ─────────────────────────────────────────────────────────────
class ToolbarManager:
    def __init__(self):
        self.visible       = False
        self.hovered_index = -1
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "#000001")
        self.root.config(bg="#000001")

        BTN_W, BTN_H, PAD, COLS = 80, 80, 15, 4
        button_defs = [
            ("Next",  ">>",  "#64dc64"),
            ("Prev",  "<<",  "#64dc64"),
            ("Zoom+", "+",   "#50b4ff"),
            ("Zoom-", "-",   "#50b4ff"),
            ("Start", "F5",  "#6464ff"),
            ("Stop",  "ESC", "#6464ff"),
            ("Draw",  "P",   "#ffc850"),
        ]
        self.buttons = []
        total_w   = COLS * BTN_W + (COLS - 1) * PAD
        total_rows = (len(button_defs) + COLS - 1) // COLS
        total_h   = total_rows * BTN_H + (total_rows - 1) * PAD
        self.start_x = (screen_w - total_w) // 2
        self.start_y = (screen_h - total_h) // 2
        self.root.geometry(f"{total_w}x{total_h}+{self.start_x}+{self.start_y}")
        self.canvas = tk.Canvas(self.root, width=total_w, height=total_h,
                                bg="#000001", highlightthickness=0)
        self.canvas.pack()

        for i, (name, icon, color) in enumerate(button_defs):
            row, col = divmod(i, COLS)
            cx = col * (BTN_W + PAD); cy = row * (BTN_H + PAD)
            r  = self.canvas.create_rectangle(cx, cy, cx+BTN_W, cy+BTN_H,
                                              fill="#333333", outline="#777777", width=2)
            ic = self.canvas.create_text(cx+BTN_W//2, cy+BTN_H//2-12,
                                         text=icon, fill="#aaaaaa",
                                         font=("Arial", 20, "bold"))
            nm = self.canvas.create_text(cx+BTN_W//2, cy+BTN_H//2+20,
                                         text=name, fill="#aaaaaa",
                                         font=("Arial", 10))
            self.buttons.append({
                "name": name, "color": color,
                "sx": self.start_x+cx, "sy": self.start_y+cy,
                "w": BTN_W, "h": BTN_H, "action": None,
                "ids": (r, ic, nm)
            })

        self._last_hov = -1
        self._last_upd = 0.0

        actions = [
            lambda: press_key(VK_RIGHT),
            lambda: press_key(VK_LEFT),
            lambda: press_key(VK_OEM_PLUS),
            lambda: press_key(VK_OEM_MINUS),
            lambda: press_key(VK_F5),
            lambda: press_key(VK_ESCAPE),
            lambda: ctrl_press_key(ord("P")),
        ]
        for i, a in enumerate(actions):
            self.buttons[i]["action"] = a

        self.root.withdraw()

    def show(self):
        self.visible = True; self.hovered_index = -1
        self.root.deiconify(); self.root.attributes("-topmost", True)

    def hide(self):
        self.visible = False; self.hovered_index = -1
        self.root.withdraw()

    def get_hovered_button(self, px: int, py: int) -> int:
        for i, b in enumerate(self.buttons):
            if (b["sx"] <= px <= b["sx"]+b["w"] and
                    b["sy"] <= py <= b["sy"]+b["h"]):
                self.hovered_index = i; return i
        self.hovered_index = -1; return -1

    def select(self):
        if 0 <= self.hovered_index < len(self.buttons):
            btn = self.buttons[self.hovered_index]
            if btn["action"]: btn["action"]()
            self.hide(); return btn["name"]
        return None

    def update(self):
        try:
            now = time.monotonic()
            if self.visible and self.hovered_index != self._last_hov:
                for i, btn in enumerate(self.buttons):
                    h  = (i == self.hovered_index)
                    bg = btn["color"] if h else "#333333"
                    fg = "#ffffff"    if h else "#aaaaaa"
                    ol = "#ffffff"    if h else "#777777"
                    lw = 4            if h else 2
                    r, ic, nm = btn["ids"]
                    self.canvas.itemconfigure(r,  fill=bg, outline=ol, width=lw)
                    self.canvas.itemconfigure(ic, fill=fg)
                    self.canvas.itemconfigure(nm, fill=fg)
                self._last_hov = self.hovered_index
            if now - self._last_upd > 0.011:
                self.root.update(); self._last_upd = now
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Gesture helpers
# ─────────────────────────────────────────────────────────────
def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def _fingers_curled(hand, margin=0.008):
    """All 4 fingers curled — tips below PIP joints."""
    return (hand[8].y  > hand[6].y  + margin and
            hand[12].y > hand[10].y + margin and
            hand[16].y > hand[14].y + margin and
            hand[20].y > hand[18].y + margin)

def is_fist(hand):
    """Fist: all 4 fingers curled + thumb tucked near palm."""
    if not _fingers_curled(hand): return False
    return dist(hand[4], hand[9]) < 0.10

def is_index_only(hand):
    """☝️ Index UP, middle DOWN, ring DOWN, pinky DOWN → next slide."""
    return (hand[8].y  < hand[6].y  and
            hand[12].y > hand[10].y and
            hand[16].y > hand[14].y and
            hand[20].y > hand[18].y)

def is_peace(hand):
    """✌️ Index UP, middle UP, ring DOWN, pinky DOWN → prev slide."""
    return (hand[8].y  < hand[6].y  and
            hand[12].y < hand[10].y and
            hand[16].y > hand[14].y and
            hand[20].y > hand[18].y)

def is_three_fingers(hand):
    """🤟 Index UP, middle UP, ring UP, pinky DOWN → start slideshow."""
    return (hand[8].y  < hand[6].y  and
            hand[12].y < hand[10].y and
            hand[16].y < hand[14].y and
            hand[20].y > hand[18].y + 0.01)

def is_open_palm(hand):
    """🖐️ All 4 fingers UP + thumb UP → zoom."""
    m = 0.01
    return (hand[8].y  < hand[6].y  - m and
            hand[12].y < hand[10].y - m and
            hand[16].y < hand[14].y - m and
            hand[20].y < hand[18].y - m and
            hand[4].y  < hand[3].y)

def is_both_index_up(rh, lh):
    """Both hands showing index-only — toolbar trigger."""
    if rh is None or lh is None: return False
    return is_index_only(rh) and is_index_only(lh)

def detect_two_fists(rh, lh):
    return (rh is not None and lh is not None and
            is_fist(rh) and is_fist(lh))

def fingers_extended(hand):
    """Index AND middle up — scroll gesture."""
    return hand[8].y < hand[6].y and hand[12].y < hand[10].y

def is_pinching(hand):
    return dist(hand[4], hand[8]) < PINCH_THRESH

def is_all_pinch(hand):
    return (dist(hand[4], hand[8]) < PINCH_THRESH and
            dist(hand[4], hand[12]) < PINCH_THRESH)

def classify_hands(result):
    """Returns (right_hand, left_hand) — MediaPipe label is mirrored."""
    right = left = None
    for i, hl in enumerate(result.handedness):
        label = hl[0].category_name
        if label == "Right": left  = result.hand_landmarks[i]
        else:                right = result.hand_landmarks[i]
    return right, left

def map_to_screen(xn, yn):
    x = max(0.0, min(1.0, (xn - MARGIN) / (1.0 - 2 * MARGIN)))
    y = max(0.0, min(1.0, (yn - MARGIN) / (1.0 - 2 * MARGIN)))
    return x * screen_w, y * screen_h


# ─────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────

# Default mode
smooth_x, smooth_y   = screen_w / 2.0, screen_h / 2.0
frozen_x, frozen_y   = screen_w / 2.0, screen_h / 2.0

lclick_down          = False
rclick_down          = False
dclick_down          = False
last_lclick_time     = 0.0
last_rclick_time     = 0.0
last_dclick_time     = 0.0
prev_scroll_y        = 0.0
scroll_active        = False

# Mode
ppt_mode             = False
ppt_toggle_start     = 0.0    # when both-fists hold began
ppt_flash_text       = ""
ppt_flash_until      = 0.0

# PPT — slideshow
slideshow_running       = False
three_finger_hold_start = 0.0
fist_stop_start         = 0.0

# PPT — toolbar
toolbar                 = ToolbarManager()
toolbar_trigger_start   = 0.0
tb_lclick_down          = False
last_tb_click_time      = 0.0

# PPT — zoom
zoom_active          = False
zoom_last_y          = 0.0
zoom_last_action_t   = 0.0

# PPT — slide navigation (index=next, peace=prev)
index_count          = 0
peace_count          = 0
slide_fired          = False   # True after gesture fired; reset on neutral
last_slide_time      = 0.0

# PPT — cursor freeze (fist-hold toggle)
cursor_user_frozen   = False
fist_freeze_start    = 0.0

# PPT — gesture debounce (prevents flickering)
prev_ppt_gesture     = "neutral"
pending_gesture      = "neutral"
pending_gesture_count = 0
GESTURE_DEBOUNCE     = 2       # frames a new gesture must hold before switching

_t0 = time.monotonic()
def get_ts(): return int((time.monotonic() - _t0) * 1000)


# ─────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────
def _bar(frame, x, y, w, h, progress, color):
    cv2.rectangle(frame, (x, y), (x + int(w * progress), y + h), color, -1)

def _txt(frame, text, x, y, scale=0.5, color=(255,255,255), thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)

def draw_hud(frame, now, rh, lh,
             three_finger_prog, fist_stop_prog, ppt_toggle_prog):
    H, W = frame.shape[:2]
    col = (0, 255, 255) if ppt_mode else (0, 255, 0)

    # ── Top centre: mode badge ──
    label = "MODE: POWERPOINT" if ppt_mode else "MODE: DEFAULT"
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cx = (W - tw) // 2
    cv2.rectangle(frame, (cx-10, 8), (cx+tw+10, 8+th+14), (0,0,0), -1)
    cv2.putText(frame, label, (cx, 8+th+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # ── Top right badges ──
    badge_y = 8
    if ppt_mode and cursor_user_frozen:
        ft = "CURSOR FROZEN"
        ftw_ = cv2.getTextSize(ft, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.rectangle(frame, (W-ftw_-20, badge_y), (W-8, badge_y+28), (0,0,200), -1)
        cv2.putText(frame, ft, (W-ftw_-14, badge_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        badge_y += 32

    # ── Top right: slideshow badge ──
    if ppt_mode and slideshow_running:
        st  = "SLIDESHOW"
        stw = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        cv2.rectangle(frame, (W-stw-20, badge_y), (W-8, badge_y+28), (0,140,0), -1)
        cv2.putText(frame, st, (W-stw-14, badge_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    # ── Zoom indicator ──
    if ppt_mode and zoom_active:
        _txt(frame, "ZOOM — move UP / DOWN", 10, 52,
             scale=0.55, color=(80,200,255), thickness=2)

    # ── Progress bars (stacked from y=58) ──
    y = 58
    if ppt_toggle_prog > 0:
        _bar(frame, 10, y, 230, 10, ppt_toggle_prog, (0,255,255))
        action = "Exit PPT..." if ppt_mode else "Enter PPT..."
        _txt(frame, action, 10, y-3, color=(0,255,255))
        y += 18

    if three_finger_prog > 0:
        _bar(frame, 10, y, 230, 10, three_finger_prog, (0,200,80))
        _txt(frame, "Hold 3-finger for slideshow...", 10, y-3, color=(0,200,80))
        y += 18

    if fist_stop_prog > 0:
        _bar(frame, 10, y, 230, 10, fist_stop_prog, (200,100,0))
        _txt(frame, "Hold to stop slideshow...", 10, y-3, color=(200,100,0))
        y += 18

    if toolbar_trigger_start > 0 and ppt_mode:
        tgt  = TOOLBAR_DISMISS_HOLD if toolbar.visible else TOOLBAR_TRIGGER_HOLD
        prog = min(1.0, (now - toolbar_trigger_start) / tgt)
        _bar(frame, 10, y, 160, 8, prog, (255,180,50))
        _txt(frame, "Toolbar...", 10, y-3, color=(255,180,50))
        y += 16

    # ── Slide confirmation counter ──
    if ppt_mode:
        if index_count > 0 and not slide_fired:
            _txt(frame, f"NEXT  {index_count}/{SLIDE_CONFIRM_FRAMES}",
                 10, y+14, scale=0.55, color=(0,255,100), thickness=2)
        if peace_count > 0 and not slide_fired:
            _txt(frame, f"PREV  {peace_count}/{SLIDE_CONFIRM_FRAMES}",
                 10, y+14, scale=0.55, color=(0,200,255), thickness=2)
        if fist_freeze_start > 0:
            fp = min(1.0, (now - fist_freeze_start) / CURSOR_FREEZE_HOLD)
            _bar(frame, 10, H-90, 160, 8, fp, (0,80,255))
            _txt(frame, "Unfreeze..." if cursor_user_frozen else "Freeze...",
                 10, H-102, scale=0.45, color=(0,80,255))

    # ── Bottom-left: hand status ──
    status = []
    if ppt_mode:
        if toolbar.visible:
            status.append("TOOLBAR OPEN")
        elif rh is not None:
            if   is_fist(rh):           status.append(f"R: FIST {'[FROZEN]' if cursor_user_frozen else ''}")
            elif is_index_only(rh):     status.append("R: INDEX")
            elif is_peace(rh):          status.append("R: PEACE")
            elif is_three_fingers(rh):  status.append("R: 3-FINGER")
            elif is_open_palm(rh):      status.append("R: PALM")
            else:                       status.append("R: ---")
        else:
            status.append("R: NO HAND")
    else:
        if rh is not None:
            rs = ("FIST" if is_fist(rh) else
                  "PINCH" if is_pinching(rh) else "MOVE")
            status.append(f"R: {rs}")
        if lh is not None:
            if lclick_down:     ls = "LEFT CLICK"
            elif rclick_down:   ls = "RIGHT CLICK"
            elif dclick_down:   ls = "DBL CLICK"
            elif scroll_active: ls = "SCROLL"
            else:               ls = "READY"
            status.append(f"L: {ls}")

    for i, txt in enumerate(status):
        cv2.putText(frame, txt, (10, H - 15 - i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    # ── Centre flash text ──
    if ppt_flash_text and now < ppt_flash_until:
        ftw = cv2.getTextSize(ppt_flash_text, cv2.FONT_HERSHEY_SIMPLEX,
                               1.2, 3)[0][0]
        fcx = (W - ftw) // 2
        cv2.rectangle(frame, (fcx-12, H//2-42), (fcx+ftw+12, H//2+20),
                      (0,0,0), -1)
        cv2.putText(frame, ppt_flash_text, (fcx, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)


# ─────────────────────────────────────────────────────────────
# Helpers — reset PPT state cleanly
# ─────────────────────────────────────────────────────────────
def _reset_ppt_state():
    global toolbar_trigger_start, tb_lclick_down
    global zoom_active, zoom_last_y, zoom_last_action_t
    global three_finger_hold_start, fist_stop_start
    global index_count, peace_count, slide_fired
    global cursor_user_frozen, fist_freeze_start
    global prev_ppt_gesture, pending_gesture, pending_gesture_count
    toolbar.hide()
    toolbar_trigger_start   = 0.0
    tb_lclick_down          = False
    zoom_active             = False
    zoom_last_y             = 0.0
    zoom_last_action_t      = 0.0
    three_finger_hold_start = 0.0
    fist_stop_start         = 0.0
    index_count             = 0
    peace_count             = 0
    slide_fired             = False
    cursor_user_frozen      = False
    fist_freeze_start       = 0.0
    prev_ppt_gesture        = "neutral"
    pending_gesture         = "neutral"
    pending_gesture_count   = 0

def _reset_default_state():
    global lclick_down, rclick_down, dclick_down
    global scroll_active, prev_scroll_y
    lclick_down   = False
    rclick_down   = False
    dclick_down   = False
    scroll_active = False
    prev_scroll_y = 0.0


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
SHOW_PREVIEW = True
cam = CameraStream(src=0, width=CAM_W, height=CAM_H)
time.sleep(0.3)

print(f"Screen: {screen_w}x{screen_h}")
print("Virtual Mouse running — press 'q' to quit")
print("Both fists held 2s → toggle PowerPoint mode")

frame_count = 0
right_hand  = None
left_hand   = None

try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # MediaPipe every 2nd frame
        if frame_count % 2 == 0:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_img, get_ts())
            right_hand, left_hand = classify_hands(result)

        now = time.monotonic()

        # ═════════════════════════════════════════════════════
        # BOTH FISTS — toggle PPT mode (always checked, top priority)
        # ═════════════════════════════════════════════════════
        ppt_toggle_prog = 0.0
        if detect_two_fists(right_hand, left_hand):
            if ppt_toggle_start == 0.0:
                ppt_toggle_start = now
            elapsed = now - ppt_toggle_start
            ppt_toggle_prog = min(elapsed / PPT_TOGGLE_HOLD, 1.0)

            if elapsed >= PPT_TOGGLE_HOLD:
                ppt_mode = not ppt_mode
                ppt_toggle_start = 0.0

                if ppt_mode:
                    # Entering PPT — freeze cursor, wipe default state
                    frozen_x, frozen_y = smooth_x, smooth_y
                    move_cursor(int(frozen_x), int(frozen_y))
                    _reset_default_state()
                    ppt_flash_text = "POWERPOINT MODE ON"
                else:
                    # Exiting PPT — wipe all PPT state
                    _reset_ppt_state()
                    ppt_flash_text = "POWERPOINT MODE OFF"

                ppt_flash_until = now + 1.5
        else:
            ppt_toggle_start = 0.0

        # ═════════════════════════════════════════════════════
        # POWERPOINT MODE
        # Right hand only. Cursor frozen unless laser active.
        # ═════════════════════════════════════════════════════
        if ppt_mode:
            rh = right_hand
            three_finger_prog = 0.0
            fist_stop_prog = 0.0

            # ── Priority 1: toolbar (both index fingers up) ──
            if is_both_index_up(right_hand, left_hand):
                if toolbar_trigger_start == 0.0:
                    toolbar_trigger_start = now
                held = now - toolbar_trigger_start
                if not toolbar.visible and held >= TOOLBAR_TRIGGER_HOLD:
                    toolbar.show()
                    toolbar_trigger_start = now
                    zoom_active = False
                elif toolbar.visible and held >= TOOLBAR_DISMISS_HOLD:
                    toolbar.hide()
                    toolbar_trigger_start = 0.0
            else:
                toolbar_trigger_start = 0.0

            # ── Toolbar open: cursor + pinch to select ──
            if toolbar.visible:
                if rh is not None:
                    raw_x, raw_y = map_to_screen(rh[8].x, rh[8].y)
                    smooth_x += SMOOTHING * (raw_x - smooth_x)
                    smooth_y += SMOOTHING * (raw_y - smooth_y)
                    move_cursor(int(smooth_x), int(smooth_y))
                    toolbar.get_hovered_button(int(smooth_x), int(smooth_y))

                    if is_pinching(rh):
                        if not tb_lclick_down and (now - last_tb_click_time) > LCLICK_COOLDOWN:
                            name = toolbar.select()
                            if name:
                                ppt_flash_text  = name
                                ppt_flash_until = now + 0.8
                                if name == "Start":   slideshow_running = True
                                elif name == "Stop":  slideshow_running = False
                            last_tb_click_time = now
                            tb_lclick_down     = True
                    else:
                        tb_lclick_down = False

            # ── Toolbar closed: gesture detection ──
            else:
                if rh is not None:

                    # ═══ FIST-FREEZE CHECK (before everything else) ═══
                    # Single right fist held 0.8s toggles cursor freeze.
                    # Does NOT conflict with fist-stop-slideshow (that
                    # only runs when slideshow_running inside the gesture
                    # block, and cursor-freeze only toggles when the
                    # slideshow is NOT running).
                    if is_fist(rh) and not detect_two_fists(right_hand, left_hand):
                        if not slideshow_running:          # freeze toggle only when no slideshow
                            if fist_freeze_start == 0.0:
                                fist_freeze_start = now
                            if (now - fist_freeze_start) >= CURSOR_FREEZE_HOLD:
                                cursor_user_frozen = not cursor_user_frozen
                                fist_freeze_start  = 0.0
                                if cursor_user_frozen:
                                    frozen_x, frozen_y = smooth_x, smooth_y
                                    ppt_flash_text  = "CURSOR FROZEN"
                                else:
                                    ppt_flash_text  = "CURSOR UNFROZEN"
                                ppt_flash_until = now + 1.0
                    else:
                        fist_freeze_start = 0.0

                    # ═══ CURSOR FROZEN → skip all gesture processing ═══
                    if cursor_user_frozen:
                        move_cursor(int(frozen_x), int(frozen_y))

                    # ═══ NORMAL GESTURE PROCESSING ═══
                    else:
                        # ─── Classify raw gesture (finger-pattern) ───
                        raw_g = "neutral"
                        if   is_fist(rh):          raw_g = "fist"
                        elif is_three_fingers(rh): raw_g = "three_fingers"
                        elif is_open_palm(rh):     raw_g = "open_palm"
                        elif is_peace(rh):         raw_g = "peace"
                        elif is_index_only(rh):    raw_g = "index"

                        # ─── Debounce ───
                        if raw_g == prev_ppt_gesture:
                            g = raw_g
                            pending_gesture = raw_g
                            pending_gesture_count = 0
                        else:
                            if raw_g == pending_gesture:
                                pending_gesture_count += 1
                            else:
                                pending_gesture = raw_g
                                pending_gesture_count = 1
                            if pending_gesture_count >= GESTURE_DEBOUNCE:
                                g = raw_g
                                prev_ppt_gesture = raw_g
                                pending_gesture_count = 0
                            else:
                                g = prev_ppt_gesture

                        # ─── Reset counters for non-active gestures ───
                        if g != "three_fingers":
                            three_finger_hold_start = 0.0
                        if g != "fist":
                            fist_stop_start = 0.0
                        if g != "index":
                            index_count = 0
                        if g != "peace":
                            peace_count = 0
                        if g not in ("index", "peace"):
                            slide_fired = False
                        if g != "open_palm":
                            zoom_active = False
                            zoom_last_y = 0.0

                        # ─── Process gesture ───

                        # Three-finger hold → start slideshow
                        if g == "three_fingers" and not slideshow_running:
                            if three_finger_hold_start == 0.0:
                                three_finger_hold_start = now
                            elapsed = now - three_finger_hold_start
                            three_finger_prog = min(elapsed / THREE_FINGER_HOLD, 1.0)
                            if elapsed >= THREE_FINGER_HOLD:
                                press_key(VK_F5)
                                slideshow_running       = True
                                three_finger_hold_start = 0.0
                                ppt_flash_text          = "SLIDESHOW STARTED"
                                ppt_flash_until         = now + 1.2

                        # Fist → stop slideshow (hold)
                        elif g == "fist":
                            # Hold for stop slideshow
                            if slideshow_running:
                                if fist_stop_start == 0.0:
                                    fist_stop_start = now
                                elapsed = now - fist_stop_start
                                fist_stop_prog = min(elapsed / FIST_STOP_HOLD, 1.0)
                                if elapsed >= FIST_STOP_HOLD:
                                    press_key(VK_ESCAPE)
                                    slideshow_running = False
                                    fist_stop_start   = 0.0
                                    ppt_flash_text    = "SLIDESHOW STOPPED"
                                    ppt_flash_until   = now + 1.2
                            move_cursor(int(frozen_x), int(frozen_y))

                        # Index only ☝️ → next slide
                        elif g == "index":
                            if not slide_fired:
                                index_count += 1
                                if (index_count >= SLIDE_CONFIRM_FRAMES and
                                        (now - last_slide_time) > PPT_SLIDE_COOLDOWN):
                                    press_key(VK_RIGHT)
                                    ppt_flash_text  = "NEXT  >"
                                    ppt_flash_until = now + 0.8
                                    last_slide_time = now
                                    slide_fired     = True
                                    index_count     = 0
                            move_cursor(int(frozen_x), int(frozen_y))

                        # Peace ✌️ → prev slide
                        elif g == "peace":
                            if not slide_fired:
                                peace_count += 1
                                if (peace_count >= SLIDE_CONFIRM_FRAMES and
                                        (now - last_slide_time) > PPT_SLIDE_COOLDOWN):
                                    press_key(VK_LEFT)
                                    ppt_flash_text  = "<  PREV"
                                    ppt_flash_until = now + 0.8
                                    last_slide_time = now
                                    slide_fired     = True
                                    peace_count     = 0
                            move_cursor(int(frozen_x), int(frozen_y))

                        # Open palm 🖐️ → zoom (Ctrl + key for reliability)
                        elif g == "open_palm":
                            wy = rh[0].y
                            if not zoom_active:
                                zoom_active        = True
                                zoom_last_y        = wy
                                zoom_last_action_t = now
                                ppt_flash_text     = "ZOOM MODE"
                                ppt_flash_until    = now + 0.5
                            else:
                                delta = wy - zoom_last_y
                                if (now - zoom_last_action_t) > ZOOM_REPEAT_DELAY:
                                    if delta < -ZOOM_VERT_DEADZONE:
                                        ctrl_press_key(VK_OEM_PLUS)
                                        ppt_flash_text     = "ZOOM +"
                                        ppt_flash_until    = now + 0.4
                                        zoom_last_action_t = now
                                        zoom_last_y        = wy
                                    elif delta > ZOOM_VERT_DEADZONE:
                                        ctrl_press_key(VK_OEM_MINUS)
                                        ppt_flash_text     = "ZOOM -"
                                        ppt_flash_until    = now + 0.4
                                        zoom_last_action_t = now
                                        zoom_last_y        = wy

                        # Neutral
                        else:
                            move_cursor(int(frozen_x), int(frozen_y))

                else:
                    # No right hand — freeze
                    index_count = peace_count = 0
                    slide_fired = False
                    zoom_active = False
                    fist_freeze_start = 0.0
                    move_cursor(int(frozen_x), int(frozen_y))

            # HUD + show
            if SHOW_PREVIEW:
                draw_hud(frame, now, right_hand, left_hand,
                         three_finger_prog, fist_stop_prog, ppt_toggle_prog)
                cv2.imshow("Virtual Mouse", frame)

        # ═════════════════════════════════════════════════════
        # DEFAULT MODE
        # Both hands active. Zero PPT logic runs here.
        # ═════════════════════════════════════════════════════
        else:

            # Right hand → cursor
            if right_hand is not None and not is_fist(right_hand):
                if not is_pinching(right_hand):
                    raw_x, raw_y = map_to_screen(right_hand[8].x, right_hand[8].y)
                    smooth_x += SMOOTHING * (raw_x - smooth_x)
                    smooth_y += SMOOTHING * (raw_y - smooth_y)
                    if (abs(smooth_x - raw_x) > DEADZONE or
                            abs(smooth_y - raw_y) > DEADZONE):
                        move_cursor(int(smooth_x), int(smooth_y))

            # Left hand → clicks + scroll
            if left_hand is not None:
                t  = left_hand[4]
                il = left_hand[8]
                ml = left_hand[12]
                d_ti = dist(t, il)
                d_tm = dist(t, ml)

                # Left click
                if d_ti < PINCH_THRESH and d_tm >= PINCH_THRESH:
                    if not lclick_down and (now - last_lclick_time) > LCLICK_COOLDOWN:
                        left_click()
                        last_lclick_time = now; lclick_down = True
                else:
                    lclick_down = False

                # Right click
                if d_tm < PINCH_THRESH and d_ti > PINCH_THRESH * 1.5:
                    if not rclick_down and (now - last_rclick_time) > RCLICK_COOLDOWN:
                        right_click()
                        last_rclick_time = now; rclick_down = True
                else:
                    rclick_down = False

                # Double click
                if is_all_pinch(left_hand):
                    if not dclick_down and (now - last_dclick_time) > DCLICK_COOLDOWN:
                        double_click()
                        last_dclick_time = now; dclick_down = True
                else:
                    dclick_down = False

                # Scroll
                if fingers_extended(left_hand) and not is_all_pinch(left_hand):
                    cy = left_hand[8].y
                    if scroll_active and prev_scroll_y != 0.0:
                        delta = (prev_scroll_y - cy) * SCROLL_SCALE
                        delta = max(-WHEEL_DELTA*5, min(WHEEL_DELTA*5, delta))
                        if abs(delta) > 8: scroll(int(delta))
                    prev_scroll_y = cy; scroll_active = True
                else:
                    prev_scroll_y = 0.0; scroll_active = False

            else:
                _reset_default_state()

            if SHOW_PREVIEW:
                draw_hud(frame, now, right_hand, left_hand,
                         0.0, 0.0, ppt_toggle_prog)
                cv2.imshow("Virtual Mouse", frame)

        toolbar.update()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()
    print("Virtual Mouse stopped.")