import json
from pathlib import Path
import logging

# Configuration
MODEL_READY_DATASET_DIR = Path("./Model_Ready_Dataset")
SEQUENCES_INFO_FILE = MODEL_READY_DATASET_DIR / "all_sequences_info.json"
LABELS_MAP = {1: "Fall", 0: "No_Fall"} # To map numerical labels back to names

def count_sequences_by_label():
    """
    Reads the sequence information JSON file and counts the number of
    sequences for each label (Fall/No_Fall).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

    if not SEQUENCES_INFO_FILE.exists():
        logging.error(f"Error: The sequence information file was not found at {SEQUENCES_INFO_FILE}")
        logging.error("Please ensure 'process_csv_data.py' has been run successfully to generate this file.")
        return

    try:
        with open(SEQUENCES_INFO_FILE, 'r') as f:
            sequences_info = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {SEQUENCES_INFO_FILE}. The file might be corrupted.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {SEQUENCES_INFO_FILE}: {e}")
        return

    if not isinstance(sequences_info, list):
        logging.error(f"Error: Expected a list in {SEQUENCES_INFO_FILE}, but got {type(sequences_info)}.")
        return

    counts = {label_name: 0 for label_name in LABELS_MAP.values()}
    
    for item in sequences_info:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            logging.warning(f"Skipping malformed item in JSON: {item}")
            continue
            
        _, label_numeric = item
        
        if label_numeric in LABELS_MAP:
            label_name = LABELS_MAP[label_numeric]
            counts[label_name] += 1
        else:
            logging.warning(f"Skipping item with unknown label: {label_numeric} in item: {item}")

    logging.info("Sequence counts in Model_Ready_Dataset:")
    for label_name, count in counts.items():
        logging.info(f"  {label_name}: {count} sequences")
    
    total_sequences = sum(counts.values())
    logging.info(f"  Total: {total_sequences} sequences")

if __name__ == "__main__":
    count_sequences_by_label()
