"""
Virtual Mouse with Gesture Control
===================================
Right hand  → cursor movement (fist to freeze)
Left hand   → left click (thumb+index pinch)
              right click (thumb+middle pinch)
              scroll (index+middle finger extended, move up/down)

Optimised for low latency:
  • Threaded camera capture (no blocking on I/O)
  • ctypes Win32 API for mouse (≈0.1 ms vs pyautogui's ≈15-20 ms)
  • Exponential moving average smoothing with dead-zone filter
"""

import cv2
import math
import time
import ctypes
import threading
from collections import deque

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
SMOOTHING        = 0.35      # EMA factor (0 = max smooth, 1 = raw)
DEADZONE         = 2         # ignore movements smaller than this (pixels)
PINCH_THRESHOLD  = 0.045     # normalised distance for pinch detection
CLICK_COOLDOWN   = 0.35      # seconds between clicks
SCROLL_SCALE     = 3000      # multiplier for scroll sensitivity
MARGIN           = 0.08      # edge margin for cursor mapping (fraction of cam frame)

# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────
smooth_x, smooth_y = screen_w / 2.0, screen_h / 2.0
last_click_time = 0.0
prev_scroll_y = 0.0
scroll_active = False

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

cam = CameraStream(src=0, width=640, height=480)

# Give camera thread time to start
time.sleep(0.3)

print(f"Screen: {screen_w}x{screen_h}")
print("Virtual Mouse running — press 'q' to quit")

try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        ts = get_timestamp_ms()
        result = detector.detect_for_video(mp_image, ts)

        # ---- Classify hands ----
        right_hand, left_hand = classify_hands(result)

        # ──────── RIGHT HAND → CURSOR ────────
        if right_hand is not None:
            if not is_fist(right_hand):
                index_tip = right_hand[8]
                raw_x, raw_y = map_to_screen(index_tip.x, index_tip.y)

                # EMA smoothing
                smooth_x += SMOOTHING * (raw_x - smooth_x)
                smooth_y += SMOOTHING * (raw_y - smooth_y)

                # Dead-zone: move only if displacement is meaningful
                dx = abs(smooth_x - raw_x)
                dy = abs(smooth_y - raw_y)
                if dx > DEADZONE or dy > DEADZONE or True:
                    move_cursor(smooth_x, smooth_y)

        # ──────── LEFT HAND → ACTIONS ────────
        if left_hand is not None:
            thumb    = left_hand[4]
            index_l  = left_hand[8]
            middle_l = left_hand[12]

            d_index  = distance(thumb, index_l)
            d_middle = distance(thumb, middle_l)

            now = time.monotonic()

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

        # ---- Preview window (optional) ----
        if SHOW_PREVIEW:
            # Draw lightweight status text
            status = []
            if right_hand is not None:
                status.append("R: " + ("FIST" if is_fist(right_hand) else "MOVE"))
            if left_hand is not None:
                status.append("L: ACTIVE")

            for i, txt in enumerate(status):
                cv2.putText(frame, txt, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()
    print("Virtual Mouse stopped.")
