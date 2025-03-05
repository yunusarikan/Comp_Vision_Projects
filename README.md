# MMORPG Automation Project

This project automates certain tasks in an MMORPG game using Python and the YOLO (You Only Look Once) model. It analyzes screen captures to detect specific objects and interacts with them using mouse and keyboard inputs.

## Features

- **Object Detection with YOLO Model**: Uses a trained YOLO model to detect specific objects (e.g., stones and HP bars) on the screen.
- **Automatic Clicking**: Automatically clicks on detected objects.
- **Forbidden Zone Control**: Skips actions if an object is detected in a forbidden zone.
- **Mouse Movement**: Moves the mouse in a specific direction to simulate in-game movement.
  
## Installation

### Requirements

- Python 3.8 or higher
- CUDA (for GPU usage)
- PyTorch
- OpenCV
- PyAutoGUI
- PyDirectInput
- Ultralytics (for YOLO)

### Installation Steps

1. **Install Python and Dependencies**:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python pyautogui pydirectinput keyboard ultralytics
Download the YOLO Model:

Place the YOLO model (best.pt) in the C:/Users/username/Desktop/VisionProject/ directory.

Run the Code:

bash
Copy
python Object_detection.py

Usage
F8: Starts the program.

F9: Stops the program.

Esc: Exits the program.

When the program starts, it analyzes screen captures and performs automated actions based on detected objects. For example, if a stone is detected, the mouse clicks on it.
