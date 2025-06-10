# --- START OF FILE split_dataset.py ---

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import shutil
import json
from sklearn.model_selection import train_test_split
import argparse
import time 

# --- Configuration ---
PROCESSED_DATA_ROOT = Path("./Model_Ready_Dataset_Padded_Interpolated")
FINAL_SPLIT_OUTPUT_ROOT = Path("./Final_Dataset_Splits")
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
TEMP_NPY_DIR_NAME = "_temp_npy_sequences"

# --- Helper Functions (if any, or can be inline) ---

def find_latest_sequences_info_file(processed_data_root: Path, process_type: str) -> Path | None:
    """Finds the latest sequence info JSON file based on process_type."""
    pattern = f"sequences_info_padded_interpolated_{process_type}_*.json"
    files = sorted(processed_data_root.glob(pattern), key=os.path.getmtime, reverse=True)
    if files:
        logging.info(f"Found sequence info file: {files[0]}")
        return files[0]
    else:
        generic_pattern = "sequences_info_padded_interpolated_both_*.json"
        files = sorted(processed_data_root.glob(generic_pattern), key=os.path.getmtime, reverse=True)
        if files:
            logging.warning(f"Specific {process_type} info file not found, using latest 'both': {files[0]}")
            return files[0]
        logging.error(f"No sequence information file matching pattern '{pattern}' or generic 'both' pattern found in {processed_data_root}.")
        return None

# --- Main Processing Logic ---
def main():
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument(
        "--process_type_for_json", type=str, default="both", choices=["fall", "no_fall", "both"],
        help="Specify which process_type's JSON info file to use for splitting. Default: 'both'."
    )
    args = parser.parse_args()

    FINAL_SPLIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_file_path = FINAL_SPLIT_OUTPUT_ROOT / f"split_dataset_{time.strftime('%Y%m%d-%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting dataset splitting and file organization...")
    logging.info(f"Source of processed data (JSON and temp .npy): {PROCESSED_DATA_ROOT.resolve()}")
    logging.info(f"Final split output will be in: {FINAL_SPLIT_OUTPUT_ROOT.resolve()}")

    sequences_info_file_path = find_latest_sequences_info_file(PROCESSED_DATA_ROOT, args.process_type_for_json)
    if not sequences_info_file_path or not sequences_info_file_path.exists():
        logging.error(f"Sequence information file could not be found. Please run create_seq.py first and ensure the file name/path is correct.")
        return

    logging.info(f"Loading sequence information from {sequences_info_file_path}...")
    try:
        with open(sequences_info_file_path, 'r') as f:
            all_sequences_info_loaded = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse {sequences_info_file_path}: {e}")
        return

    if not all_sequences_info_loaded:
        logging.error("No sequences found in the information file. Exiting.")
        return

    all_sequences_info = [(Path(item[0]), item[1]) for item in all_sequences_info_loaded]
    logging.info(f"Successfully loaded {len(all_sequences_info)} sequence entries.")

    logging.info("Splitting data into train, validation, and test sets...")
    sequence_paths_from_json = [info[0] for info in all_sequences_info]
    sequence_labels = [info[1] for info in all_sequences_info]

    if len(set(sequence_labels)) < 2 and len(sequence_labels) > 0:
        logging.warning("Only one class found in the data. Stratified split will behave like random split or fail if too few samples.")
    elif not sequence_labels:
         logging.error("No labels found for splitting. Exiting.")
         return

    try:
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            sequence_paths_from_json, sequence_labels,
            test_size=SPLIT_RATIOS["test"],
            random_state=42,
            stratify=sequence_labels if len(set(sequence_labels)) > 1 else None
        )

        if SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"] == 0:
            val_relative_size = 0
        else:
            val_relative_size = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])

        if train_val_paths:
             can_stratify_second_split = len(set(train_val_labels)) > 1 and len(train_val_paths) > 1
             if not can_stratify_second_split and len(train_val_paths) > 1:
                  logging.warning("Only one class in train/val for second split or too few samples. Using non-stratified.")
            
             train_paths, val_paths, train_labels, val_labels = train_test_split(
                 train_val_paths, train_val_labels,
                 test_size=val_relative_size if val_relative_size > 0 and len(train_val_paths) * val_relative_size >=1 else 0, # Ensure test_size is valid
                 random_state=42,
                 stratify=train_val_labels if can_stratify_second_split else None
             )
        else:
             train_paths, val_paths, train_labels, val_labels = [], [], [], []
             logging.warning("Train+Validation set is empty after first split.")

        split_data = {
            "train": list(zip(train_paths, train_labels)),
            "val": list(zip(val_paths, val_labels)),
            "test": list(zip(test_paths, test_labels))
        }
        logging.info(f"Split sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    except ValueError as e:
         logging.error(f"Splitting failed. This can happen with very small datasets or severe class imbalance. Error: {e}")
         logging.warning("Assigning all sequences to 'train' set as fallback.")
         split_data = {
            "train": list(zip(sequence_paths_from_json, sequence_labels)),
            "val": [],
            "test": []
         }

    logging.info("Organizing files into final output directories and creating summary...")
    summary_data = []
    
    for split_name, sequences_in_split in split_data.items():
        logging.info(f"Organizing {split_name} set...")
        if not sequences_in_split:
            logging.info(f"No sequences for {split_name} set. Skipping.")
            continue
        for source_npy_path, label in sequences_in_split:
            target_label_dir_name = "fall" if label == 1 else "no_fall"
            target_split_label_dir = FINAL_SPLIT_OUTPUT_ROOT / split_name / target_label_dir_name
            target_split_label_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_split_label_dir / source_npy_path.name
            
            try:
                if not source_npy_path.exists():
                    logging.error(f"Source file {source_npy_path} (from JSON) does not exist. Skipping this file.")
                    continue
                shutil.copy2(str(source_npy_path), str(target_path))
                logging.debug(f"Copied {source_npy_path.name} to {target_path}")
                summary_data.append({
                    "filename": target_path.name,
                    "label": label,
                    "split": split_name,
                    "final_path_relative": str(target_path.relative_to(FINAL_SPLIT_OUTPUT_ROOT))
                })
            except FileNotFoundError:
                 logging.error(f"Source file {source_npy_path} not found during copy. Skipping.")
            except Exception as e:
                logging.error(f"Failed to copy {source_npy_path.name} to {target_path}: {e}")

    temp_npy_sequences_dir = PROCESSED_DATA_ROOT / TEMP_NPY_DIR_NAME
    logging.info(f"Checking temporary directory for cleanup: {temp_npy_sequences_dir}")
    if temp_npy_sequences_dir.exists() and temp_npy_sequences_dir.is_dir():
        logging.info(f"Temporary directory {temp_npy_sequences_dir} exists. "
                     f"Consider manual deletion after verifying all files are copied to {FINAL_SPLIT_OUTPUT_ROOT}.")
    else:
        logging.info(f"Temporary directory {temp_npy_sequences_dir} does not exist or is not a directory.")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = FINAL_SPLIT_OUTPUT_ROOT / "dataset_summary.csv"
        try:
            summary_df.to_csv(summary_csv_path, index=False)
            logging.info(f"Final dataset summary created at: {summary_csv_path}")
        except Exception as e:
            logging.error(f"Failed to create final dataset summary CSV: {e}")
    else:
        logging.warning("No data was successfully copied to create a summary.csv.")

    logging.info("Dataset splitting and file organization finished.")

if __name__ == "__main__":
    main()
# --- END OF FILE split_dataset.py ---