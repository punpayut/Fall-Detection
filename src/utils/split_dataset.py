import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import shutil
import json # For loading sequence info
from sklearn.model_selection import train_test_split

# --- Configuration ---
OUTPUT_ROOT_DIR = Path("./Model_Ready_Dataset")
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15} # Ensure these sum to 1.0
SEQUENCES_INFO_FILE = OUTPUT_ROOT_DIR / "all_sequences_info.json" # File to load sequence paths and labels
TEMP_NPY_DIR_NAME = "_temp_npy" # Name of the temporary directory for .npy files

# --- Main Processing Logic ---

def main():
    # --- Configure Logging ---
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True) # Ensure output root exists for log file
    log_file_path = OUTPUT_ROOT_DIR / "split_dataset.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
            logging.StreamHandler() # To also print to console
        ]
    )
    # --- End Logging Configuration ---

    logging.info("Starting dataset splitting and file organization...")

    # 1. Load sequence information
    logging.info(f"Loading sequence information from {SEQUENCES_INFO_FILE}...")
    if not SEQUENCES_INFO_FILE.exists():
        logging.error(f"Sequence information file not found: {SEQUENCES_INFO_FILE}. Run process_csv_data.py first.")
        return

    try:
        with open(SEQUENCES_INFO_FILE, 'r') as f:
            all_sequences_info_loaded = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse {SEQUENCES_INFO_FILE}: {e}")
        return

    if not all_sequences_info_loaded:
        logging.error("No sequences found in the information file. Exiting.")
        return

    # Convert string paths back to Path objects
    all_sequences_info = [(Path(item[0]), item[1]) for item in all_sequences_info_loaded]
    logging.info(f"Successfully loaded {len(all_sequences_info)} sequence entries.")

    # 2. Split Data
    logging.info("Splitting data into train, validation, and test sets...")
    sequence_paths = [info[0] for info in all_sequences_info]
    sequence_labels = [info[1] for info in all_sequences_info]

    if len(set(sequence_labels)) < 2:
        logging.warning("Only one class found in the data. Stratified split may behave like random split.")
        if not sequence_labels:
             logging.error("No labels found for splitting. Exiting.")
             return

    try:
        # First split: Train + Val vs Test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            sequence_paths, sequence_labels,
            test_size=SPLIT_RATIOS["test"],
            random_state=42,
            stratify=sequence_labels
        )

        val_relative_size = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])

        if train_val_paths:
             if len(set(train_val_labels)) < 2 and len(train_val_paths) > 1 : # Check if stratification is possible
                  logging.warning("Only one class found in train/val set for second split. Using non-stratified split.")
                  train_paths, val_paths, train_labels, val_labels = train_test_split(
                       train_val_paths, train_val_labels,
                       test_size=val_relative_size,
                       random_state=42
                  )
             elif not train_val_paths: # Should not happen if train_val_paths is checked before
                  train_paths, val_paths, train_labels, val_labels = [], [], [], []
             else: # Stratification is possible or only one sample
                  train_paths, val_paths, train_labels, val_labels = train_test_split(
                       train_val_paths, train_val_labels,
                       test_size=val_relative_size,
                       random_state=42,
                       stratify=train_val_labels if len(train_val_paths) > 1 else None # Stratify only if more than 1 sample
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
         logging.error(f"Splitting failed. Maybe too few samples per class? Error: {e}")
         logging.warning("Assigning all sequences to 'train' set as fallback due to splitting error.")
         split_data = {
            "train": list(zip(sequence_paths, sequence_labels)),
            "val": [],
            "test": []
         }

    # 3. Organize Files into Output Structure and Create Summary
    logging.info("Organizing files into output directories and creating summary...")
    summary_data = []
    temp_npy_dir = OUTPUT_ROOT_DIR / TEMP_NPY_DIR_NAME # Define temp_npy_dir based on OUTPUT_ROOT_DIR

    for split_name, sequences in split_data.items():
        logging.info(f"Organizing {split_name} set...")
        if not sequences:
            logging.info(f"No sequences for {split_name} set. Skipping.")
            continue
        for npy_path, label in sequences:
            target_label_dir = "fall" if label == 1 else "no_fall"
            target_dir = OUTPUT_ROOT_DIR / split_name / target_label_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            # npy_path is already a Path object from the loaded data
            source_npy_path = npy_path
            # Resolve the source path to ensure it's absolute and fully qualified
            resolved_source_path = source_npy_path.resolve()
            target_path = target_dir / source_npy_path.name
            
            try:
                # Check existence using the resolved path
                if not resolved_source_path.exists():
                    logging.error(f"Source file {resolved_source_path} does not exist. It might have been moved or deleted, or path in JSON is incorrect. Skipping this file.")
                    continue

                # Use shutil.copy2() for copying
                shutil.copy2(str(resolved_source_path), str(target_path))
                logging.info(f"Successfully copied {resolved_source_path.name} to {target_path}")
                summary_data.append({
                    "filename": target_path.name, # Store relative to the split/label dir
                    "label": label,
                    "split": split_name,
                    "original_path": str(source_npy_path), # For debugging if needed
                    "final_path": str(target_path)
                })
            except FileNotFoundError:
                 logging.error(f"Source file {resolved_source_path} not found during copy attempt. Skipping this file.")
            except Exception as e:
                logging.error(f"Failed to copy {resolved_source_path.name} (source: {resolved_source_path}) to {target_path}: {e}")

    # 4. Remove temporary directory if it's empty
    logging.info(f"Attempting to clean up temporary directory: {temp_npy_dir}")
    try:
        if temp_npy_dir.exists():
            # Check if it's truly empty. If files failed to move, it might not be.
            is_empty = not any(temp_npy_dir.iterdir())
            if is_empty:
                 temp_npy_dir.rmdir()
                 logging.info(f"Successfully removed empty temporary directory: {temp_npy_dir}")
            else:
                 logging.warning(f"Temporary directory {temp_npy_dir} is not empty. Some files may not have been moved. Manual cleanup might be needed.")
        else:
            logging.info(f"Temporary directory {temp_npy_dir} does not exist. No cleanup needed.")
    except OSError as e:
        logging.warning(f"Could not remove temporary directory {temp_npy_dir}: {e}")

    # 5. Create summary.csv
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = OUTPUT_ROOT_DIR / "summary.csv"
        try:
            summary_df.to_csv(summary_csv_path, index=False)
            logging.info(f"Summary file created at: {summary_csv_path}")
        except Exception as e:
            logging.error(f"Failed to create summary CSV: {e}")
    else:
        logging.warning("No data to write to summary.csv. Skipping summary file creation.")

    logging.info("Dataset splitting and file organization finished.")

if __name__ == "__main__":
    main()
