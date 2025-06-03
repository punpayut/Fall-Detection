# AI Fall Detection Gradio App on Hugging Face Spaces

This application provides a web interface using Gradio to detect falls in uploaded video files. It utilizes MediaPipe for pose estimation and a TFLite Transformer model for fall classification.

## Features

*   **User-friendly Web Interface:** Upload videos directly through your browser.
*   **Fall Detection:** Analyzes videos frame-by-frame to detect potential fall events.
*   **Pose Estimation:** Visualizes detected human poses (skeletons) using MediaPipe.
*   **Processed Video Output:** Returns the input video with overlaid pose skeletons and detection status.
*   **Detection Summary:** Provides a textual summary of detected events and processing status.
*   **Example Videos:** Includes pre-set example videos for quick testing.
*   **Optimized for Spaces:** `pose_complexity` is set lower for faster processing on shared resources.

## Files Required in Your Hugging Face Space Repository

To successfully deploy this on Hugging Face Spaces, ensure the following files are present in the **root** of your repository:

1.  **`app.py`**: The main Gradio application script (the code you provided).
2.  **`requirements.txt`**: Lists the Python dependencies.
    ```text
    gradio
    opencv-python-headless
    mediapipe
    numpy
    tflite-runtime
    ```
3.  **`fall_detection_transformer.tflite`**: The TensorFlow Lite model file for fall detection.
4.  **Example Video Files (Optional but Recommended):**
    The `app.py` script looks for these example files in the root directory. If you want the examples to work, include them:
    *   `fall_example_1.mp4`
    *   `fall_example_2.mp4`
    *   `fall_example_3.mp4`
    *   `fall_example_4.mp4`
    *(Ensure the filenames match exactly those specified in the `example_filenames` list within `app.py`)*

## Deployment to Hugging Face Spaces

1.  **Create a Hugging Face Account:** If you don't have one, sign up at [huggingface.co](https://huggingface.co/).
2.  **Create a New Space:**
    *   Click on your profile picture, then "New Space".
    *   Choose an **Owner** (your username or an organization).
    *   Give your Space a **Name** (e.g., `fall-detector-gradio`).
    *   Select a **License** (e.g., `apache-2.0`).
    *   Choose **Gradio** as the Space SDK.
    *   Choose the **Space hardware**. The free "CPU basic" tier should work for testing, but processing will be slower.
    *   Select whether the Space is **Public** or **Private**.
    *   Click "Create Space".
3.  **Upload Files to the Space:**
    *   Once the Space is created, go to the "Files" tab.
    *   You can upload files one by one using the "Upload file" button or clone the repository locally, add files, commit, and push.
    *   Ensure `app.py`, `requirements.txt`, `fall_detection_transformer.tflite`, and your example video files are all in the **root directory** of the Space repository.
4.  **Building the Space:**
    *   Hugging Face Spaces should automatically detect `requirements.txt` and `app.py` (since it's a Gradio SDK Space) and start building the environment.
    *   You can monitor the build process in the "Logs" tab.
    *   If it doesn't build automatically, or if you make changes, you might need to manually trigger a rebuild from the Space settings.
5.  **Access Your App:**
    *   Once the build is successful and the app is running, you'll be able to access the Gradio interface directly from the Space's main page (the "App" tab).

## How to Use the Application

1.  **Navigate to your Hugging Face Space URL.**
2.  You will see the Gradio interface titled "AI Fall Detection from Video".
3.  **Upload a Video:**
    *   Drag and drop an MP4 video file into the "Upload Video File" box, or click to browse your local files.
4.  **Or Use Example Videos:**
    *   Click on one of the provided example videos listed below the upload box. This will automatically load and process the selected example.
5.  **Processing:**
    *   The application will start processing the video. This may take some time depending on the video's length, resolution, and the current load on the Hugging Face Spaces hardware.
    *   A "Detection Summary" will update with frame processing status.
6.  **View Results:**
    *   **Processed Video:** Once complete, a video player will appear showing the original video with pose landmarks and detection status text overlaid.
    *   **Detection Summary:** A textbox will display key events, such as detected falls and their approximate timestamps in the video.

## Configuration Notes

*   **`pose_complexity`**: In `app.py`, this is set to `0` for faster processing on Hugging Face Spaces. If you have more powerful dedicated hardware, you could consider increasing it to `1` for potentially better pose accuracy (at the cost of speed).
*   **`FALL_CONFIDENCE_THRESHOLD`**: This is set to `0.90` in `app.py`. You can adjust this value if you want to be more or less sensitive to fall detections.
*   **Model Input Shape:** The script checks if the `NUM_FEATURES` and `INPUT_TIMESTEPS` match the loaded TFLite model's expectations. Ensure your model `fall_detection_transformer.tflite` is compatible with the keypoint definitions and sequence length defined in `app.py`.

## Troubleshooting

*   **"File not found" for example videos:** Ensure the example video files (`fall_example_1.mp4`, etc.) are present in the **root** of your Hugging Face Space repository and their names exactly match those in the `example_filenames` list in `app.py`.
*   **Slow Processing:** Video processing, especially on the free CPU tier of Hugging Face Spaces, can be slow for longer or high-resolution videos. Be patient or try with shorter clips for testing.
*   **App Crashes / Errors:** Check the "Logs" tab in your Hugging Face Space for error messages. This can help diagnose issues related to missing files, package incompatibilities, or code errors.
*   **Model Errors:** If you see errors related to the TFLite model, ensure `fall_detection_transformer.tflite` is not corrupted and its input/output specifications match what `app.py` expects (particularly `NUM_FEATURES` and `INPUT_TIMESTEPS`).