# 🎯 Smart Focus Assistant

This project implements a **Real-Time Posture & Attention Monitoring System** using computer vision and MediaPipe. By analyzing your body posture and gaze direction through a webcam, it provides live feedback to help you stay focused and ergonomically healthy during study or work sessions.

## 🚀 Project Overview

*   **Goal**: Detect slouching and attention loss in real time to promote healthy, productive work habits.
*   **Input**: Live webcam feed.
*   **Models**: MediaPipe Face Landmarker, Pose Landmarker, and Selfie Segmenter.
*   **Interface**: Annotated OpenCV window with live posture and attention status overlays.

## 🧠 How It Works

*   **Posture Detection**: Measures the vertical distance between ear and shoulder landmarks. If the ratio drops below a calibrated threshold, a slouching warning is triggered.
*   **Attention / Gaze Tracking**: Uses facial landmark geometry (nose, chin, eyes) to estimate yaw (left/right head turn) and pitch (up/down tilt). Deviations beyond defined thresholds flag the user as distracted.
*   **Background Segmentation**: Applies a selfie segmentation mask to blur the background, keeping the focus on the subject.
*   **Smoothing**: Exponential Moving Average (EMA) is applied to all metrics to eliminate flickering and noisy readings.
*   **Dynamic Calibration**: Press `C` at any time to recalibrate the slouch threshold to your current sitting posture (30-frame average).

## 🛠️ Tech Stack

*   **Language**: Python 3
*   **Computer Vision**: OpenCV
*   **Landmark Detection**: MediaPipe Tasks (Face Landmarker, Pose Landmarker, Image Segmenter)
*   **Numerical Processing**: NumPy

## 📂 Project Structure

*   `main.py`: Core application — initializes all three MediaPipe models, processes the webcam feed, and renders the live overlay.
*   `face_landmarker.task`: Pre-trained MediaPipe face landmark detection model.
*   `pose_landmarker.task`: Pre-trained MediaPipe pose landmark detection model.
*   `selfie_segmenter.tflite`: Pre-trained TFLite selfie segmentation model.
*   `requirements.txt`: Project dependencies.

## ⚙️ How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/Youssef-Mohamed4/smart-focus-assistant.git
cd smart-focus-assistant
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the assistant**:
```bash
py main.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `C` | Calibrate slouch threshold to current posture |
| `ESC` or `X` | Exit the application |

## 🔧 Tunable Parameters

These thresholds can be adjusted at the top of `main.py` to suit your setup:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `global_slouch_threshold` | `0.28` | Ear-to-shoulder ratio below which slouching is flagged |
| `GAZE_TURN_LEFT` | `0.82` | Yaw lower bound for centered gaze |
| `GAZE_TURN_RIGHT` | `1.25` | Yaw upper bound for centered gaze |
| `PITCH_LOOK_UP` | `0.35` | Pitch lower bound for centered gaze |
| `PITCH_LOOK_DOWN` | `0.90` | Pitch upper bound for centered gaze |