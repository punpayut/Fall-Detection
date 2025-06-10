import os
import cv2
import mediapipe as mp
import csv
from pathlib import Path

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose

# Define the indices for the desired 17 keypoints.
KEYPOINT_INDICES = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def extract_17_keypoints(video_path, output_csv):
    # Create a MediaPipe Pose object.
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    frame_idx = 1  # Start frame index from 1

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Keypoint", "X", "Y", "Confidence"])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for idx, kp_name in zip(KEYPOINT_INDICES, KEYPOINT_NAMES):
                    lm = landmarks[idx]
                    x_pixel = lm.x * frame.shape[1]
                    y_pixel = lm.y * frame.shape[0]
                    writer.writerow([frame_idx, kp_name, x_pixel, y_pixel, lm.visibility])
            
            frame_idx += 1

    cap.release()
    pose.close()
    # print(f"Finished processing video, closed resources.") # Optional: Can be verbose

def process_dataset(root_dir):
    # Convert string path to Path object
    root_dir = Path(root_dir)
    print(f"[Info] Starting dataset processing in root: {root_dir}")

    # Iterate over main categories: Fall and No_Fall.
    for category in ['Fall', 'No_Fall']:
        category_path = root_dir / category
        print(f"[Info] Checking category path: {category_path}")
        if not category_path.exists():
            print(f"[Warning] Category directory not found, skipping: {category_path}")
            continue

        # Process subcategories: Raw_Video.
        for subcat in ['Raw_Video']:
            subcat_path = category_path / subcat
            print(f"[Info] Checking subcategory path: {subcat_path}")
            if not subcat_path.exists():
                print(f"[Warning] Subcategory directory not found, skipping: {subcat_path}")
                continue

            # Create an output directory for CSV files.
            csv_output_dir = category_path / 'Keypoints_CSV'
            print(f"[Info] Ensuring output directory exists: {csv_output_dir}")
            try:
                csv_output_dir.mkdir(parents=True, exist_ok=True) # Added parents=True
            except OSError as e:
                print(f"[Error] Failed to create directory {csv_output_dir}: {e}. Skipping this subcategory.")
                continue # Skip this subcategory if dir creation fails

            # Process each video file in the subcategory folder.
            video_files_found = False
            print(f"[Info] Looking for videos (.mp4, .avi, .mov) in: {subcat_path}")
            for file_path in subcat_path.glob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                    video_files_found = True
                    output_csv = csv_output_dir / f"{file_path.stem}_keypoints.csv"
                    print(f"[Info] Found video: {file_path}. Target CSV: {output_csv}")

                    # Check if the CSV file already exists
                    if output_csv.exists():
                        print(f"[Info] Skipping {file_path.name}, CSV already exists: {output_csv}")
                        continue  # Skip to the next video file

                    # Proceed with extraction if CSV does not exist
                    print(f"[Info] Processing video to: {output_csv}")
                    try:
                        extract_17_keypoints(str(file_path), str(output_csv))
                        # Check if the file was created and has significant content
                        if output_csv.exists():
                            if output_csv.stat().st_size > 100: # Check size > header (approx)
                                print(f"[Success] Keypoints saved to: {output_csv} (Size: {output_csv.stat().st_size} bytes)")
                            else:
                                print(f"[Warning] Output CSV file exists but is small (possibly empty or header only): {output_csv} (Size: {output_csv.stat().st_size} bytes)")
                        else:
                             print(f"[Error] Output CSV file was not created: {output_csv}")
                    except Exception as e:
                        print(f"[Error] Exception during extract_17_keypoints for {file_path}: {e}")

            if not video_files_found:
                 print(f"[Warning] No video files (.mp4, .avi, .mov) found in {subcat_path}")

if __name__ == "__main__":
    # Define project root explicitly to avoid path calculation issues observed
    # Assumes the project structure is c:/src/projects/AI-Builder/
    # with 'src' and 'Data_Repository' inside it.
    project_root_path = Path("./")

    # The dataset root IS the project root in this case, as Fall/No_Fall are directly inside it.
    dataset_root_path = project_root_path

    print(f"[Setup] Script location: {Path(__file__).resolve()}") # Still print actual script location
    print(f"[Setup] Assumed project root: {project_root_path}") # This is also the dataset root now
    print(f"[Setup] Target dataset root: {dataset_root_path}")

    if not dataset_root_path.exists() or not dataset_root_path.is_dir():
        print(f"[Error] Dataset root directory not found or not a directory at: {dataset_root_path}")
        print("Please ensure 'Data_Repository' directory exists at the project root level (adjacent to the 'src' directory).")
    else:
        print(f"[Info] Dataset root found. Starting processing...")
        try:
            process_dataset(dataset_root_path) # Pass the Path object
        except Exception as e:
            print(f"[Critical Error] An unexpected error occurred during dataset processing: {e}")
    print("[Info] Script finished.")
