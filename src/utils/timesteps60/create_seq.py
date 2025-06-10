# --- START OF FILE create_seq.py ---
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json
import argparse

# Import from common_keypoints
try:
    from common_keypoints import KEYPOINT_ORDER_ALPHABETICAL, NUM_KEYPOINTS as COMMON_NUM_KEYPOINTS
except ImportError:
    print("CRITICAL ERROR: common_keypoints.py not found. This script cannot run without it.")
    exit()

# --- Configuration ---
INPUT_ROOT_DIR = Path("./")
OUTPUT_ROOT_DIR = Path("./Model_Ready_Dataset_Padded_Interpolated")
SEQUENCE_LENGTH = 60
STEP_SIZE_FALL = 12
STEP_SIZE_NO_FALL = 30
LABELS = {"Fall": 1, "No_Fall": 0}

NUM_KEYPOINTS = COMMON_NUM_KEYPOINTS
NUM_FEATURES_PER_KEYPOINT = 3
EXPECTED_FEATURES_PER_FRAME = NUM_KEYPOINTS * NUM_FEATURES_PER_KEYPOINT

# --- Logging Setup (ย้ายมาไว้ก่อนเพื่อให้ KEYPOINT logging แสดงผล) ---
OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
temp_log_filename_for_type_detection = "temp_arg_parse_log.txt"
parser_temp = argparse.ArgumentParser(add_help=False)
parser_temp.add_argument("--process_type", type=str, default="both", choices=["fall", "no_fall", "both"])
args_temp, _ = parser_temp.parse_known_args()

log_file_name = f"create_seq_{args_temp.process_type}_{time.strftime('%Y%m%d-%H%M%S')}.log"
log_file_path = OUTPUT_ROOT_DIR / log_file_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()]
)

logging.info(f"--- Initializing Keypoint Order for .npy Output ---")
logging.info(f"Using KEYPOINT_ORDER_ALPHABETICAL for .npy structure: {KEYPOINT_ORDER_ALPHABETICAL}")
if NUM_KEYPOINTS != 17:
    logging.error(f"CRITICAL: NUM_KEYPOINTS from common_keypoints.py is {NUM_KEYPOINTS}, expected 17.")
    exit()
if EXPECTED_FEATURES_PER_FRAME != 51:
    logging.error(f"CRITICAL: EXPECTED_FEATURES_PER_FRAME is {EXPECTED_FEATURES_PER_FRAME}, expected 51.")
    exit()
logging.info("---------------------------------------------------")

# --- Helper Functions ---
def process_csv_for_sequencing(csv_path: Path, sequence_length: int, step_size: int, category_name: str):
    sequences = []
    sequence_file_info = []
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.warning(f"CSV file is empty: {csv_path}. Skipping.")
            return [], []

        required_cols = ['Frame', 'Keypoint', 'X', 'Y', 'Confidence']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Missing one or more essential columns in {csv_path}. Skipping file.")
            return [], []

        for col_to_check in ['Frame', 'X', 'Y', 'Confidence']:
            if not pd.api.types.is_numeric_dtype(df[col_to_check]):
                try:
                    df[col_to_check] = pd.to_numeric(df[col_to_check], errors='coerce')
                    if df[col_to_check].isnull().any():
                         logging.warning(f"NaNs introduced in '{col_to_check}' during to_numeric for {csv_path}.")
                except ValueError:
                    logging.error(f"Could not convert '{col_to_check}' to numeric in {csv_path}. Skipping file.")
                    return [],[]
        
        if df['Keypoint'].nunique() == 0:
            logging.warning(f"No unique keypoints found to pivot in {csv_path}. Skipping file.")
            return [], []

        frame_features = df.pivot_table(
            index='Frame', columns='Keypoint', values=['X', 'Y', 'Confidence']
        )
        frame_features.columns = ['_'.join(map(str, col)) for col in frame_features.columns.values]

        if df['Frame'].empty or df['Frame'].min() > df['Frame'].max():
            logging.warning(f"No valid frame range in {csv_path}. Skipping.")
            return [], []

        min_frame = int(df['Frame'].min())
        max_frame = int(df['Frame'].max())
        all_frames_index = pd.RangeIndex(start=min_frame, stop=max_frame + 1, name='Frame')

        # --- Define expected columns in ALPHABETICAL order using imported KEYPOINT_ORDER_ALPHABETICAL ---
        expected_cols_alphabetical_ordered = []
        for kp_name in KEYPOINT_ORDER_ALPHABETICAL: # Use imported ALPHABETICAL order
            expected_cols_alphabetical_ordered.append(f"X_{kp_name}")
            expected_cols_alphabetical_ordered.append(f"Y_{kp_name}")
            expected_cols_alphabetical_ordered.append(f"Confidence_{kp_name}")
        
        # Reindex with all frames and ensure all expected columns are present and in ALPHABETICAL order
        frame_features = frame_features.reindex(index=all_frames_index, columns=expected_cols_alphabetical_ordered)

        # --- Interpolation for X and Y (iterating based on ALPHABETICAL order) ---
        for kp_name in KEYPOINT_ORDER_ALPHABETICAL:
            x_col = f"X_{kp_name}"
            y_col = f"Y_{kp_name}"
            if x_col in frame_features.columns:
                frame_features[x_col] = frame_features[x_col].interpolate(method='linear', limit_direction='both', limit_area='inside')
            if y_col in frame_features.columns:
                frame_features[y_col] = frame_features[y_col].interpolate(method='linear', limit_direction='both', limit_area='inside')
        
        frame_features = frame_features.fillna(0.0)

        if frame_features.shape[1] != EXPECTED_FEATURES_PER_FRAME:
             logging.critical(f"FATAL: Feature count mismatch for {csv_path} AFTER processing. Expected {EXPECTED_FEATURES_PER_FRAME}, got {frame_features.shape[1]}. Columns: {frame_features.columns.tolist()}")
             return [], []

        data = frame_features.values
        num_frames = data.shape[0]

        if num_frames == 0:
            logging.warning(f"No data rows after all processing for {csv_path}. Skipping.")
            return [], []

        base_name_for_file = csv_path.stem.replace("_keypoints", "")

        if num_frames < sequence_length:
            if num_frames > 0:
                padding_needed = sequence_length - num_frames
                last_frame_data = data[-1, :].reshape(1, -1)
                padding_data = np.repeat(last_frame_data, padding_needed, axis=0)
                padded_sequence = np.vstack((data, padding_data))
                sequences.append(padded_sequence)
                sequence_file_info.append((base_name_for_file, csv_path.name, 0, category_name.lower()))
                logging.debug(f"Padded {csv_path.name}: {num_frames} to {sequence_length} frames.")
            else:
                logging.warning(f"Skipping padding for {csv_path.name} as it has 0 effective frames.")
        else:
            seq_idx = 0
            for i in range(0, num_frames - sequence_length + 1, step_size):
                seq = data[i : i + sequence_length]
                sequences.append(seq)
                sequence_file_info.append((base_name_for_file, csv_path.name, seq_idx, category_name.lower()))
                seq_idx += 1
            if seq_idx == 0:
                logging.info(f"No full sequences from sliding window for {csv_path.name} (Frames: {num_frames}, SeqLen: {sequence_length}, Step: {step_size})")
        return sequences, sequence_file_info
    except pd.errors.EmptyDataError:
        logging.warning(f"CSV file {csv_path} is empty or invalid. Skipping.")
        return [], []
    except Exception as e:
        logging.error(f"Failed to process CSV {csv_path}: {e}", exc_info=True)
        return [], []

# --- Main Processing Logic ---
def main():
    parser = argparse.ArgumentParser(description="Process CSV keypoint data into ALPHABETICALLY ordered sequences with padding and interpolation.")
    parser.add_argument(
        "--process_type", type=str, choices=["fall", "no_fall", "both"], default="both",
        help="Type of data to process: 'fall', 'no_fall', or 'both'. Default is 'both'."
    )
    args = parser.parse_args() # Parsed after logging is setup, but this is fine.

    logging.info(f"--- Configuration ---")
    logging.info(f"Processing type: {args.process_type}")
    logging.info(f"Input Root: {INPUT_ROOT_DIR.resolve()}")
    logging.info(f"Output Root: {OUTPUT_ROOT_DIR.resolve()} (Keypoints in .npy will be ALPHABETICAL)")
    logging.info(f"Sequence Length: {SEQUENCE_LENGTH} frames")
    logging.info(f"Step Size (Fall): {STEP_SIZE_FALL}")
    logging.info(f"Step Size (No_Fall): {STEP_SIZE_NO_FALL}")
    logging.info("Padding short videos with last frame. Interpolating missing X,Y.")
    logging.info("--- End Configuration ---")

    categories_to_process = {}
    if args.process_type == "fall":
        categories_to_process = {"Fall": LABELS["Fall"]}
    elif args.process_type == "no_fall":
        categories_to_process = {"No_Fall": LABELS["No_Fall"]}
    elif args.process_type == "both":
        categories_to_process = LABELS
    else:
        logging.error(f"Invalid process_type: {args.process_type}")
        return

    logging.info("Pre-calculating expected number of sequences for each category...")
    expected_sequence_counts = {}
    all_potential_file_actions = []

    for category, label in categories_to_process.items():
        category_input_path = INPUT_ROOT_DIR / category / "Keypoints_CSV"
        expected_sequence_counts[category] = 0
        if not category_input_path.is_dir():
            logging.warning(f"Pre-calc: Directory not found: {category_input_path}")
            continue
        csv_files = list(category_input_path.glob("*.csv"))
        if not csv_files:
            logging.warning(f"Pre-calc: No CSV files found in: {category_input_path}")
            continue
        
        current_category_file_infos = []
        for csv_file in csv_files:
            try:
                df_temp = pd.read_csv(csv_file, usecols=['Frame'])
                if df_temp.empty or 'Frame' not in df_temp.columns or df_temp['Frame'].nunique() == 0 :
                    logging.warning(f"Pre-calc: Skipping empty or invalid frame data in {csv_file.name}")
                    continue
                num_unique_frames = df_temp['Frame'].nunique()
                current_category_file_infos.append((csv_file, category, label, num_unique_frames))
                current_step_size = STEP_SIZE_FALL if category == "Fall" else STEP_SIZE_NO_FALL
                if num_unique_frames < SEQUENCE_LENGTH:
                    if num_unique_frames > 0: 
                        expected_sequence_counts[category] += 1
                else:
                    if num_unique_frames >= SEQUENCE_LENGTH:
                        num_possible_seqs = (num_unique_frames - SEQUENCE_LENGTH) // current_step_size + 1
                        expected_sequence_counts[category] += num_possible_seqs
            except Exception as e:
                logging.warning(f"Pre-calc: Error lightly reading {csv_file} for counting: {e}")
        all_potential_file_actions.extend(current_category_file_infos)
    
    logging.info("--- Expected Sequence Counts (Approximate) ---")
    total_expected_sequences = 0
    for category, count in expected_sequence_counts.items():
        logging.info(f"Category '{category}': {count} sequences")
        total_expected_sequences += count
    logging.info(f"Total expected sequences: {total_expected_sequences}")
    logging.info("---------------------------------------------")

    if total_expected_sequences == 0:
        logging.warning("No sequences are expected to be generated. Exiting.")
        return

    proceed = input("Do you want to proceed with generating these .npy files? (y/n): ").strip().lower()
    if proceed != 'y':
        logging.info("User chose not to proceed. Exiting.")
        return

    logging.info("Proceeding with .npy file generation...")
    all_sequences_info_list_for_json = []
    base_temp_npy_dir = OUTPUT_ROOT_DIR / "_temp_npy_sequences" # Generic name for temp dir
    base_temp_npy_dir.mkdir(parents=True, exist_ok=True)

    processed_csv_files_count = 0
    saved_sequences_count = 0

    for csv_file_path, category_name, label, _ in all_potential_file_actions:
        category_output_npy_dir = base_temp_npy_dir / category_name
        category_output_npy_dir.mkdir(parents=True, exist_ok=True)
        current_step_size_for_file = STEP_SIZE_FALL if category_name == "Fall" else STEP_SIZE_NO_FALL
        logging.info(f"Processing: {csv_file_path.name} [{category_name}]")
        # Removed expected_features argument as it's now a global constant based on NUM_KEYPOINTS
        video_sequences, generated_file_infos = process_csv_for_sequencing(
            csv_file_path, SEQUENCE_LENGTH, current_step_size_for_file, category_name
        )

        if not video_sequences:
            logging.warning(f"No sequences generated for {csv_file_path.name} during full processing.")
            continue
        processed_csv_files_count +=1

        for seq_data, (base_name, _, seq_idx, cat_name_lower) in zip(video_sequences, generated_file_infos):
            if seq_data.shape != (SEQUENCE_LENGTH, EXPECTED_FEATURES_PER_FRAME):
                 logging.error(f"Sequence {seq_idx} for {base_name} from {csv_file_path.name} has INCORRECT SHAPE: {seq_data.shape}. Expected: {(SEQUENCE_LENGTH, EXPECTED_FEATURES_PER_FRAME)}. Skipping save.")
                 continue
            npy_filename = f"{base_name}_{cat_name_lower}_seq_{seq_idx:03d}.npy"
            npy_path = category_output_npy_dir / npy_filename
            try:
                np.save(npy_path, seq_data.astype(np.float32))
                logging.debug(f"Saved: {npy_path} (Shape: {seq_data.shape}) - Keypoints ordered alphabetically.")
                all_sequences_info_list_for_json.append((str(npy_path), label))
                saved_sequences_count +=1
            except Exception as e:
                logging.error(f"Failed to save {npy_path}: {e}", exc_info=True)

    logging.info(f"Finished processing. Successfully processed {processed_csv_files_count} CSV files that yielded sequences.")
    logging.info(f"Total .npy sequences saved: {saved_sequences_count}")
    
    sequences_info_file_name = f"sequences_info_padded_interpolated_{args.process_type}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    sequences_info_file_path = OUTPUT_ROOT_DIR / sequences_info_file_name
    try:
        with open(sequences_info_file_path, 'w') as f:
            json.dump(all_sequences_info_list_for_json, f, indent=4)
        logging.info(f"Successfully saved sequence information to {sequences_info_file_path}")
    except Exception as e:
        logging.error(f"Failed to save sequence information to {sequences_info_file_path}: {e}")

    logging.info(f"CSV processing and sequence generation finished.")

if __name__ == "__main__":
    main()
# --- END OF FILE create_seq.py ---