import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json # For saving sequence info

# --- Configuration ---
INPUT_ROOT_DIR = Path("./")  # Assumes script is run from the directory containing Fall/No_Fall
OUTPUT_ROOT_DIR = Path("./Model_Ready_Dataset")
SEQUENCE_LENGTH = 30
# STEP_SIZE = 15 # Will be defined per category
STEP_SIZE_FALL = 3
STEP_SIZE_NO_FALL = 6
NUM_KEYPOINTS = 17
NUM_FEATURES_PER_KEYPOINT = 3 # X, Y, Confidence
EXPECTED_FEATURES_PER_FRAME = NUM_KEYPOINTS * NUM_FEATURES_PER_KEYPOINT # 17 * 3 = 51
LABELS = {"Fall": 1, "No_Fall": 0}
SEQUENCES_INFO_FILE = OUTPUT_ROOT_DIR / "all_sequences_info.json" # File to store sequence paths and labels

# --- Helper Functions ---

def process_csv(csv_path: Path, sequence_length: int, step_size: int, expected_features: int):
    """Reads a CSV, reshapes data, creates overlapping sequences, and indicates if padding occurred."""
    file_was_padded = False
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.warning(f"CSV file is empty: {csv_path}")
            return [], file_was_padded

        frame_features = df.pivot_table(
            index='Frame', 
            columns='Keypoint', 
            values=['X', 'Y', 'Confidence']
        )
        
        frame_features.columns = ['_'.join(map(str, col)).strip() for col in frame_features.columns.values]
        
        all_frames = range(int(df['Frame'].min()), int(df['Frame'].max()) + 1)
        frame_features = frame_features.reindex(all_frames)
        frame_features = frame_features.fillna(0) 

        if frame_features.shape[1] != expected_features:
             logging.warning(f"Frame features count mismatch in {csv_path}. Expected {expected_features}, got {frame_features.shape[1]}. Skipping file.")
             return [], file_was_padded

        data = frame_features.values
        num_frames = data.shape[0]
        sequences = []

        for i in range(0, num_frames - sequence_length + 1, step_size):
            seq = data[i : i + sequence_length]
            sequences.append(seq)

        last_start_index = (num_frames - sequence_length) // step_size * step_size
        remaining_frames = num_frames - (last_start_index + step_size)

        if remaining_frames > 0 and num_frames >= sequence_length :
             last_seq_start = last_start_index + step_size
             last_seq = data[last_seq_start:]
             padding_length = sequence_length - last_seq.shape[0]
             if padding_length > 0:
                 padding = np.zeros((padding_length, expected_features))
                 last_seq = np.vstack((last_seq, padding))
                 file_was_padded = True # Padding occurred
             if last_seq.shape[0] == sequence_length:
                 sequences.append(last_seq)
             else:
                  logging.warning(f"Error padding last sequence in {csv_path}. Shape: {last_seq.shape}. Skipping last partial sequence.")
        elif num_frames < sequence_length and num_frames > 0:
            logging.info(f"Video shorter than sequence length: {csv_path}. Padding single sequence.")
            seq = data
            padding_length = sequence_length - seq.shape[0]
            padding = np.zeros((padding_length, expected_features))
            seq = np.vstack((seq, padding))
            file_was_padded = True # Padding occurred
            if seq.shape[0] == sequence_length:
                 sequences.append(seq)
            else:
                 logging.warning(f"Error padding short video sequence in {csv_path}. Shape: {seq.shape}. Skipping.")
        return sequences, file_was_padded

    except pd.errors.EmptyDataError:
        logging.warning(f"CSV file is empty or invalid: {csv_path}")
        return [], file_was_padded
    except Exception as e:
        logging.error(f"Failed to process CSV {csv_path}: {e}")
        return [], file_was_padded

# --- Main Processing Logic ---

def main():
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = OUTPUT_ROOT_DIR / "process_csv_data.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting CSV processing and sequence generation...")
    all_sequences_info_list = [] # List to store tuples: (npy_path_str, label)
    padded_file_count = 0

    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    temp_npy_dir = OUTPUT_ROOT_DIR / "_temp_npy"
    temp_npy_dir.mkdir(exist_ok=True)

    logging.info("Processing CSV files and creating sequences...")
    for category, label in LABELS.items():
        category_path = INPUT_ROOT_DIR / category / "Keypoints_CSV"
        logging.info(f"Processing category: {category} (Label: {label})")
        if not category_path.is_dir():
            logging.warning(f"Directory not found: {category_path}")
            continue

        csv_files = list(category_path.glob("*.csv"))
        if not csv_files:
            logging.warning(f"No CSV files found in: {category_path}")
            continue
            
        logging.info(f"Found {len(csv_files)} CSV files in {category_path}.")

        for csv_file in csv_files:
            logging.info(f"Processing: {csv_file.name}")
            
            current_step_size = STEP_SIZE_FALL if category == "Fall" else STEP_SIZE_NO_FALL
            logging.info(f"Using STEP_SIZE = {current_step_size} for category '{category}'")

            video_sequences, file_was_padded = process_csv(csv_file, SEQUENCE_LENGTH, current_step_size, EXPECTED_FEATURES_PER_FRAME)
            
            if file_was_padded:
                padded_file_count += 1

            if not video_sequences:
                logging.warning(f"No sequences generated for {csv_file.name}.")
                continue

            base_name = csv_file.stem.replace("_keypoints", "")
            for i, seq in enumerate(video_sequences):
                if seq.shape != (SEQUENCE_LENGTH, EXPECTED_FEATURES_PER_FRAME):
                     logging.error(f"Sequence {i} for {base_name} has incorrect shape: {seq.shape}. Expected: {(SEQUENCE_LENGTH, EXPECTED_FEATURES_PER_FRAME)}. Skipping save.")
                     continue
                 
                npy_filename = f"{base_name}_seq_{i:03d}.npy"
                npy_path = temp_npy_dir / npy_filename
                logging.info(f"Attempting to save sequence {i} for {base_name} to {npy_path}. Shape: {seq.shape}, Type: {seq.dtype}")
                try:
                    np.save(npy_path, seq)
                    logging.info(f"Successfully saved sequence {i} for {base_name} to {npy_path}")
                    if npy_path.exists():
                        logging.info(f"Verified: File {npy_path} exists immediately after save.")
                    else:
                        logging.error(f"Verification FAILED: File {npy_path} does NOT exist immediately after np.save reported success.")
                    # Store string representation of path for JSON serialization
                    all_sequences_info_list.append((str(npy_path), label)) 
                    #time.sleep(0.1) 
                except Exception as e:
                    logging.error(f"Failed to save sequence {i} for {base_name} to {npy_path}: {e}", exc_info=True)

    if not all_sequences_info_list:
        logging.error("No sequences were generated or saved. Exiting.")
        return

    logging.info(f"Total sequences generated: {len(all_sequences_info_list)}")
    logging.info(f"Total number of CSV files that required padding: {padded_file_count}")

    # Save all_sequences_info to a JSON file
    try:
        with open(SEQUENCES_INFO_FILE, 'w') as f:
            json.dump(all_sequences_info_list, f, indent=4)
        logging.info(f"Successfully saved sequence information to {SEQUENCES_INFO_FILE}")
    except Exception as e:
        logging.error(f"Failed to save sequence information to {SEQUENCES_INFO_FILE}: {e}")

    logging.info("CSV processing and sequence generation finished.")

if __name__ == "__main__":
    main()
