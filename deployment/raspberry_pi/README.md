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

*   **"Error: Cannot open webcam"**:
    *   Ensure your webcam is properly connected and recognized by the Raspberry Pi. You can check available video devices by running:
        ```bash
        ls /dev/video*
        ```
    *   Check the `CAMERA_INDEX` constant in `fall-detector.py` if you have multiple cameras (e.g., `0` for the first, `1` for the second).
    *   Make sure no other application is currently using the webcam.
    *   **Permissions**: Your user might need to be part of the `video` group. Add your user to the group with:
        ```bash
        sudo usermod -a -G video $USER
        ```
        You'll need to log out and log back in for this change to take effect.

*   **Low FPS / Slow Performance**:
    *   Running on a Raspberry Pi 5 is highly recommended for better performance.
    *   Ensure your Raspberry Pi has an adequate power supply and is not overheating (which can cause thermal throttling).
    *   The `pose_complexity` variable within the `fall-detector.py` script (default is `1`) can be set to `0`. This will make MediaPipe's pose estimation faster but potentially less accurate.
        ```python
        # Inside fall-detector.py, near the MediaPipe Pose initialization
        pose_complexity = 0 # Faster, less accurate
        # pose_complexity = 1 # Default
        # pose_complexity = 2 # Slower, more accurate
        ```
    *   Consider reducing the `FRAME_WIDTH` and `FRAME_HEIGHT` constants in `fall-detector.py` to process smaller images, which can improve speed.

*   **`ImportError: ...` (e.g., for `cv2`, `mediapipe`, `tflite_runtime`)**:
    *   Make sure you have activated the Python virtual environment before running the script:
        ```bash
        source .venv/bin/activate
        ```
    *   If the error persists, try reinstalling the problematic package within the activated virtual environment:
        ```bash
        pip uninstall <package_name>
        pip install <package_name>
        ```
        For example:
        ```bash
        pip uninstall opencv-python
        pip install opencv-python
        ```

*   **TensorFlow Lite Model Error ("Model's expected input shape ... does not match ...")**:
    *   Verify that the `MODEL_PATH` constant in `fall-detector.py` points to the correct TFLite model file.
    *   Crucially, ensure that the model's expected input shape (number of timesteps and number of features per timestep) matches the configuration in the script. The script has built-in checks and will print an error message if there's a mismatch.
        *   `INPUT_TIMESTEPS` in the script must match the first dimension (after batch) of the model's input.
        *   `NUM_FEATURES` (derived from `NUM_KEYPOINTS_TRAINING * 3`) must match the second dimension of the model's input.
        *   Adjust `INPUT_TIMESTEPS` or review `KEYPOINT_DICT_TRAINING` and `SORTED_YOUR_KEYPOINT_NAMES` in `fall-detector.py` if these values don't align with your trained model.

*   **Telegram messages not being sent**:
    *   Double-check the `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` values in your `.env` file. Ensure there are no extra spaces or typos.
    *   Verify that `ENABLE_TELEGRAM_ALERTS` is set to `True` in `fall-detector.py`.
    *   Confirm that your Raspberry Pi has an active and stable internet connection.
    *   Inspect the `fall_detection_log.txt` file for any error messages related to Telegram API requests (e.g., "Error sending Telegram message: ...").
    *   If you recently created the bot or started a chat, ensure you've sent at least one message *to* the bot from your Telegram account so it can identify your `CHAT_ID`.

*   **Python script does not run or shows permission denied**:
    *   Ensure the script file has execute permissions:
        ```bash
        chmod +x fall-detector.py
        ```
    *   Then try running with `python3 fall-detector.py` or `./fall-detector.py` (if you add a shebang like `#!/usr/bin/env python3` to the top of the script).
