# Fall Detection Model Notebooks

This repository contains Jupyter notebooks exploring different models for fall detection based on human pose keypoint sequences.

## Notebook Summaries

### 1. `notebook/fall-detection-bilstm-f1-bl.ipynb` -- 😄 Use for base line

*   **Objective:** Classify sequences of human pose keypoints (30 frames x 51 features) as 'fall' or 'no_fall'.
*   **Data:** Loads pre-processed `.npy` keypoint sequences from train/validation/test splits. Normalizes X/Y coordinates.
*   **Model:** Uses a **Bidirectional LSTM (BiLSTM)** network with two BiLSTM layers and dropout.
*   **Training:** Employs **Focal Loss** and **class weights** to handle potential class imbalance. Uses Adam optimizer, early stopping, learning rate reduction, and a custom callback to save the model based on the best **macro F1 score** on the validation set.
*   **Evaluation:** Reports detailed metrics (Precision, Recall, F1, F2) for both validation and test sets, saves them to CSV, and visualizes results with bar plots and confusion matrices.

### 2. `notebook/fall-detection-lstm-d3-v1.ipynb` -- ▶️ 1st model training with various dataset

*   **Objective:** Classify sequences of human pose keypoints (30 frames x 51 features) as 'fall' or 'no_fall'.
*   **Data:** Uses the same data loading and normalization process as the BiLSTM notebook.
*   **Model:** Uses a standard **LSTM** network (non-bidirectional) with two LSTM layers and dropout.
*   **Training:** Employs standard **binary cross-entropy loss** and the Adam optimizer. Uses early stopping based on validation loss to prevent overfitting (does *not* use focal loss or class weights). Saves the model based on the best **validation loss**.
*   **Evaluation:** Reports detailed metrics (Precision, Recall, F1, F2) for both validation and test sets, saves them to CSV, and visualizes results with bar plots and confusion matrices.

## Key Differences

*   **Model Architecture:** BiLSTM vs. standard LSTM.
*   **Loss Function:** Focal Loss (BiLSTM) vs. Binary Cross-Entropy (LSTM).
*   **Class Imbalance Handling:** Class weights used (BiLSTM) vs. not used (LSTM).
*   **Best Model Selection:** Based on Macro F1 score (BiLSTM) vs. Validation Loss (LSTM).
