# Gesture-Controlled Virtual Mouse (v4)

A smart, real-time virtual mouse application that uses a webcam to track hand gestures for controlling the Windows mouse cursor, clicks, scrolling, and presentation features (specifically optimized for Microsoft PowerPoint). Powered by **OpenCV** and Google's **MediaPipe Hand Landmarker**.

---

## 🚀 Features & Control Modes

This application operates in two distinct modes: **Default Mouse Mode** and **PowerPoint Mode**.

### 1. Default Mouse Mode (Standard OS Interaction)
In this mode, you can control the desktop mouse cursor and execute clicks/scrolls using both hands:

- **Right Hand 🫱 (Cursor Movement)**:
  - **Move Cursor**: Move your right hand in the camera view to control the cursor.
  - **Freeze Cursor (Fist ✊)**: Ball your right hand into a fist to temporarily freeze cursor movement.
- **Left Hand 🫲 (Clicks & Scrolls)**:
  - **Left Click**: Pinch your thumb + index finger (🤏).
  - **Right Click**: Pinch your thumb + middle finger.
  - **Double Click**: Pinch thumb + index + middle fingers together.
  - **Scroll**: Extend index and middle fingers together, then move your hand vertically to scroll up or down.

---

### 2. PowerPoint Mode (Presentation Tools)
Designed specifically for presentations. In this mode, the left hand is ignored, and the right hand controls slide transitions, slideshow options, and zoom:

*To **Toggle PowerPoint Mode ON/OFF**, hold **both fists (✊ + ✊) for 2 seconds**.*

| Gesture / Action | Description |
| :--- | :--- |
| **Both Fists (✊ ✊) for 2s** | Toggle PowerPoint Mode ON or OFF |
| **Both Indexes Up (☝️ ☝️) for 0.3s** | Show/Hide the interactive floating toolbar |
| **Three Fingers (🤟) for 1s** | Start PowerPoint Slideshow (`F5`) |
| **Fist (✊) for 1.5s** | Stop PowerPoint Slideshow (`Esc`) |
| **Index Finger Up (☝️)** | Next Slide |
| **Peace Sign (✌️)** | Previous Slide |
| **Open Palm (🖐️) + Move Vertically** | Zoom In / Zoom Out |
| **Fist (✊) for 0.8s** | Lock/Unlock cursor position |

---

## 🛠️ Installation & Setup

### Prerequisites
- Windows OS (uses Win32 API calls for cursor/keyboard interaction).
- Python 3.8 or higher.
- A functional webcam.

### 1. Clone the Repository
```bash
git clone https://github.com/NisargPatel007/Virtual-mouse.git
cd Virtual-mouse
```

### 2. Install Dependencies
Install the required Python modules:
```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe Model
The app requires the MediaPipe Hand Landmarker model file (`hand_landmarker.task`) to run. If it is not present, download it from the following link and place it in the project root directory:

👉 **[Download hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)**

---

## 🎮 How to Run

Launch the application:
```bash
python main.py
```

- A HUD camera window will appear showing your camera feed, current mode badge, detected hands, and progress bars.
- Press **`q`** while focusing on the camera window to safely exit the application.
