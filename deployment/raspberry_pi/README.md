# Real-time Fall Detection System on Raspberry Pi 5

This project implements a real-time fall detection system using a TFLite Transformer model, MediaPipe for pose estimation, and OpenCV for video processing. It is designed to run on a Raspberry Pi 5 and can send alerts via Telegram.

## Features

*   Real-time pose estimation using MediaPipe Pose.
*   Fall detection using a custom TFLite Transformer model.
*   Sequential feature processing (30 timesteps).
*   Skeleton normalization for improved model robustness.
*   Optional Telegram alerts for detected falls.
*   Supports webcam or video file input.
*   Configuration via `.env` file and command-line arguments.
*   Logging of events.

## Prerequisites

### Hardware
*   Raspberry Pi 5 (A Raspberry Pi 4 Model B with 4GB+ RAM might also work, but performance may vary)
*   Webcam compatible with Raspberry Pi (e.g., USB webcam, Raspberry Pi Camera Module)
*   Power supply for Raspberry Pi 5
*   MicroSD card (16GB or larger, Class 10 recommended)
*   (Optional) Monitor, keyboard, mouse for setup

### Software
*   **Raspberry Pi OS (64-bit)**: Recommended for best compatibility and performance with MediaPipe and TensorFlow Lite. (e.g., "Raspberry Pi OS with desktop (64-bit)")
*   **Python 3.9+**: Typically pre-installed on recent Raspberry Pi OS versions. You can check with `python3 --version`.
*   **pip**: Python package installer. Usually comes with Python.
*   **Git**: For cloning this repository.

## Installation

1.  **Clone the Repository:**
    Open a terminal on your Raspberry Pi and run:
    ```bash
    git clone <your-repository-url>
    cd <repository-name> # e.g., cd fall-detector
    ```

2.  **Set up a Python Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(To deactivate later, simply type `deactivate`)*

3.  **Install System Dependencies (if needed for OpenCV):**
    OpenCV might require some system libraries. Most should be present on a desktop version of Raspberry Pi OS, but if you encounter issues during `pip install`, you might need:
    ```bash
    sudo apt update
    sudo apt install -y libopencv-dev python3-opencv # This installs system OpenCV, pip might still build its own
    sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test # For older systems or if pip fails
    ```
    However, try installing with `pip` first, as it often handles dependencies better.

4.  **Install Python Dependencies:**
    Make sure your virtual environment is activated.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `opencv-python` and `mediapipe` on Raspberry Pi can take some time.*

5.  **Download/Place the TFLite Model:**
    Ensure the TFLite model file `fall_detection_transformer.tflite` is present in the same directory as `fall-detector.py`. If you have it elsewhere, update `MODEL_PATH` in the script or move the file.

6.  **Set up Telegram Alerts (Optional):**
    If you want to receive Telegram notifications:
    *   **Create a Telegram Bot:**
        1.  Open Telegram and search for "BotFather".
        2.  Start a chat with BotFather and send `/newbot`.
        3.  Follow the instructions to choose a name and username for your bot.
        4.  BotFather will give you an **API token**. Copy this token.
    *   **Get Your Chat ID:**
        1.  Search for your newly created bot in Telegram and send it a message (e.g., `/start`).
        2.  Open a web browser and go to the following URL, replacing `<YOUR_BOT_TOKEN>` with the token you copied:
            `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
        3.  Look for the `result` array. Inside, find an object with a `message` key. Inside `message`, find `chat` and then `id`. This is your `CHAT_ID`.
    *   **Create `.env` file:**
        In the same directory as `fall-detector.py`, create a file named `.env` with the following content:
        ```env
        TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
        TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID"
        ```
        Replace the placeholder values with your actual bot token and chat ID.
    *   Ensure `ENABLE_TELEGRAM_ALERTS` is `True` in `fall-detector.py` (it is by default).

## Configuration

The script can be configured via:

*   **Constants at the top of `fall-detector.py`:**
    *   `MODEL_PATH`: Path to the TFLite model.
    *   `INPUT_TIMESTEPS`: Sequence length for the model.
    *   `FALL_CONFIDENCE_THRESHOLD`: Threshold for fall detection.
    *   `MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION`: Confidence for keypoints used in normalization.
    *   `ENABLE_TELEGRAM_ALERTS`: Set to `False` to disable Telegram.
    *   `CAMERA_INDEX`: `0` for default webcam, `1` for the next, etc.
    *   `FRAME_WIDTH`, `FRAME_HEIGHT`: Desired camera resolution.
    *   `LOG_FILE`: Name of the log file.
    *   `FALL_EVENT_COOLDOWN`: Cooldown period (seconds) between fall alerts.
*   **`.env` file:**
    *   `TELEGRAM_BOT_TOKEN`: Your Telegram bot API token.
    *   `TELEGRAM_CHAT_ID`: Your Telegram chat ID.
*   **Command-line arguments:**
    *   `--source`: `webcam` (default) or `file`.
    *   `--file`: Path to video file if `source` is `file`.

## Usage

Ensure your virtual environment is activated (`source .venv/bin/activate`).

### Using Webcam:
```bash
python fall-detector.py
```
Or explicitly:
```bash
python fall-detector.py --source webcam
```

A window titled "Fall Detection Monitor" should appear showing the camera feed with pose landmarks and status.

## Using a Video File:
```bash
python fall-detector.py --source file --file /path/to/your/video.mp4
```
Replace `/path/to/your/video.mp4` with the actual path to your video file.

## Exiting the Program
Press `q` in the "Fall Detection Monitor" window to stop the script.

## Logging
The script logs events (start, stop, fall detections, errors) to fall_detection_log.txt by default.

## Troubleshooting
**"Error: Cannot open webcam":**
Ensure your webcam is properly connected and recognized by the Raspberry Pi (ls /dev/video* should list devices).
Check CAMERA_INDEX in the script if you have multiple cameras.
Make sure no other application is using the webcam.
Permissions: Your user might need to be in the video group: sudo usermod -a -G video $USER (logout and login again after this).
Low FPS / Slow Performance:
Running on a Raspberry Pi 5 is recommended.
Ensure the Pi is adequately powered and cooled.
The pose_complexity in the script (default 1) can be set to 0 for faster but less accurate pose estimation.
Reduce FRAME_WIDTH and FRAME_HEIGHT for the camera.
ImportError: ... (e.g., for cv2 or mediapipe):
Ensure you have activated the virtual environment (source .venv/bin/activate).
Try reinstalling the problematic package: pip uninstall <package_name> then pip install <package_name>.
TensorFlow Lite Model Error:
Verify MODEL_PATH is correct.
Ensure the model input shape matches the script's configuration (INPUT_TIMESTEPS, NUM_FEATURES). The script has checks for this.
Telegram messages not sent:
Double-check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.
Ensure ENABLE_TELEGRAM_ALERTS is True in the script.
Verify the Raspberry Pi has an active internet connection.
Check fall_detection_log.txt for any error messages related to Telegram.
