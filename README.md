# Fall Detection System

:smiley: This project provides an integrated solution for real-time fall detection using computer vision. It is designed to monitor individuals and provide immediate notifications when a fall is detected, making it suitable for applications involving elderly or vulnerable individuals.

## Overview

The system leverages MediaPipe for robust human pose estimation and a custom-trained TensorFlow Lite (TFLite) Transformer model for accurate fall classification. The project includes a comprehensive data processing pipeline, various model training notebooks, and deployment options for both web-based (Hugging Face Spaces) and edge device (Raspberry Pi) environments.

## Demo

![Fall Detection Demo](/images/fall_detection_demo.gif)

## Features

*   **Real-time Fall Detection:** Utilizes advanced computer vision techniques to identify fall events as they occur.
*   **Pose Estimation:** Integrates MediaPipe Pose for precise human keypoint extraction.
*   **Transformer-based Classification:** Employs a TFLite Transformer model for high-accuracy fall detection from sequential pose data.
*   **Flexible Data Processing:** Scripts for extracting keypoints from raw videos, processing them into fixed-length sequences (30 or 60 frames), and splitting datasets for model training.
*   **Multiple Deployment Options:**
    *   **Web Application:** A Gradio-based web interface deployable on Hugging Face Spaces for easy video upload and analysis.
    *   **Edge Device:** An optimized Python application for real-time monitoring on Raspberry Pi 5, including optional Telegram alerts.
*   **Comprehensive Logging & Configuration:** Detailed logging and configurable parameters for various components of the system.

## Methodology

The methodology involves extracting 17 pose keypoints per frame using MediaPipe Pose, normalizing the data, and creating 30 or 60-frame sequences with augmentations. The pipeline spans from feature extraction to model-ready sequences.

## Models

The project evaluated multiple deep learning architectures for classifying human fall sequences:
*   **LSTM (Long Short-Term Memory):** Learns temporal patterns using gated memory units.
*   **Bi-LSTM (Bidirectional Long Short-Term Memory):** Enhances context by processing sequences in both forward and backward directions.
*   **Transformer:** Captures long-range dependencies with attention mechanisms and positional encoding.

## Evaluation Metrics

The models were evaluated using the following metrics:
*   **F1-Macro Score:** Averages F1-scores across all classes for balanced evaluation, even when fall cases are rare.
*   **Recall (Fall Class Only):** Focuses on catching real falls by minimizing false negatives, which are critical to safety.

## Results

Transformer models significantly outperformed LSTM and Bi-LSTM, especially in detecting falls. The best performance was achieved by the Transformer with 60-frame inputs and data augmentation, reaching 94.9% F1-score and 94.1% recall for the critical fall class. This model showed the lowest false negatives—critical for real-world safety applications—and significantly outperforms LSTM and Bi-LSTM baselines.

## Discussion & Conclusion

This research successfully demonstrates the superior performance of Transformer-based architectures for temporal pose analysis in fall detection. The system enables real-time inference with high accuracy and is optimized for edge deployment on Raspberry Pi, making it practical for elderly care applications. A complete working prototype with alert mechanisms and a public demo is available for community validation.

## Future Work

Future work will focus on multi-person fall detection, expanding dataset diversity, and improving robustness.

## Project Structure

The repository is organized into the following main directories:

*   `dataset/`: Contains information and links to the raw video datasets and extracted keypoints used for training.
*   `src/`: Houses the core source code, including:
    *   `src/utils/`: Python scripts for the data processing pipeline (keypoint extraction, sequence creation, dataset splitting).
    *   `src/utils/timesteps60/`: Specialized utility scripts for processing 60-frame sequences.
    *   `src/notebook/`: Jupyter Notebooks detailing the model training and experimentation process on Kaggle, exploring various architectures like LSTM and Transformer.
*   `deployment/`: Contains scripts and configurations for deploying the fall detection system:
    *   `deployment/huggingface_space/`: Files required to deploy the Gradio web application on Hugging Face Spaces.
    *   `deployment/raspberry_pi/`: Scripts and instructions for setting up and running the real-time fall detection system on a Raspberry Pi 5.
*   `images/`: Stores project-related images, including the demo GIF.

## Dataset

The project utilizes datasets consisting of raw video files and their corresponding extracted human pose keypoints in CSV format. Two primary sequence lengths are supported: 30 frames and 60 frames. The datasets are typically split into training, validation, and testing sets with a ratio of 70:15:15 (%).

*   [Raw video (MP4) and extracted keypoints (CSV) dataset](https://www.kaggle.com/datasets/payutch/fall-video-dataset)
*   [30-frame sequence dataset](https://kaggle.com/datasets/6e0c174777418da58bcdd22bbfe5f93c551fda84d13da7662320eef320c7ba84)
*   [60-frame sequence dataset](https://kaggle.com/datasets/56c36a8a7481518e5ad56e155e6030fddcbabffa7f7a8902e496fb9c483c85a7)

## Source Code (`src/`)

The `src` directory contains the Python scripts and Jupyter notebooks essential for data preparation, model training, and experimentation.

### `src/utils/`

This directory contains a set of Python scripts to process video data for the fall detection system. The pipeline involves:

1.  **`extract_keypoint.py`**: Extracts 17 human body keypoints from raw video files using MediaPipe Pose and saves them into CSV format.
2.  **`process_csv_data.py`**: Processes the extracted keypoint CSV files into fixed-length sequences (default 30 frames) suitable for model training. It handles missing frames, creates overlapping sequences, and saves them as `.npy` files and a JSON info file.
3.  **`split_dataset.py`**: Splits the generated sequences into training, validation, and test sets and organizes them into a structured directory.
4.  **`count_processed_sequences.py`**: Counts the number of processed sequences for each class (Fall/No_Fall).

### `src/utils/timesteps60/`

This subdirectory contains utility scripts specifically tailored for processing 60-frame sequences:

1.  **`common_keypoints.py`**: Defines shared constants and configurations for human body keypoints and skeleton connections.
2.  **`extract_keypoint.py`**: (Similar to `src/utils/extract_keypoint.py`) Extracts 17 keypoints from videos to CSV.
3.  **`create_seq.py`**: Processes CSVs into 60-frame `.npy` sequences, handling interpolation and padding, and generates a JSON info file.
4.  **`display_seq.py`**: Loads a single `.npy` sequence file and visualizes the skeleton animation, saving it as a GIF.
5.  **`display_skeleton.py`**: Reads keypoint data directly from a CSV file, visualizes the skeleton animation, and saves it as a GIF.
6.  **`split_dataset_v2.py`**: Splits the 60-frame `.npy` sequences into training, validation, and test sets.

### `src/notebook/`

This directory contains Jupyter Notebooks used for training and experimenting with different fall detection models on Kaggle. These notebooks explore various architectures, including LSTM and Transformer models, and demonstrate the training process using the prepared keypoint sequence datasets.

## Deployment (`deployment/`)

The `deployment` directory provides solutions for deploying the fall detection system in different environments.

### Hugging Face Spaces (`deployment/huggingface_space/`)

This application provides a web interface using Gradio to detect falls in uploaded video files. It utilizes MediaPipe for pose estimation and a TFLite Transformer model for fall classification.

**Features:**
*   User-friendly Web Interface for video uploads.
*   Analyzes videos frame-by-frame to detect fall events.
*   Visualizes detected human poses (skeletons).
*   Returns processed video with overlaid pose skeletons and detection status.
*   Provides a textual summary of detected events.
*   Includes pre-set example videos for quick testing.

**Deployment:**
To deploy on Hugging Face Spaces, ensure `app.py`, `requirements.txt`, `fall_detection_transformer.tflite`, and example video files are in the root of your Space repository. Detailed deployment and usage instructions are available in `deployment/huggingface_space/README.md`.

### Raspberry Pi 5 (`deployment/raspberry_pi/`)

This project implements a real-time fall detection system designed to run on a Raspberry Pi 5. It uses a TFLite Transformer model, MediaPipe for pose estimation, and OpenCV for video processing, with optional Telegram alerts.

**Features:**
*   Real-time pose estimation and fall detection.
*   Supports webcam or video file input.
*   Sequential feature processing (30 timesteps).
*   Skeleton normalization for improved model robustness.
*   Optional Telegram alerts for detected falls.
*   Configurable via `.env` file and command-line arguments.
*   Event logging.

**Prerequisites:**
Raspberry Pi 5, compatible webcam, Python 3.9+, pip, and Git.

**Installation & Usage:**
Detailed instructions for installation, setting up Telegram alerts, configuration, and usage are provided in `deployment/raspberry_pi/README.md`.
