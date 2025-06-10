import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from collections import defaultdict

def display_skeleton_from_csv(csv_path):
    frames_data = defaultdict(list)
    keypoint_order = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]
    keypoint_map = {name: i for i, name in enumerate(keypoint_order)}
    num_coco_keypoints = len(keypoint_order)

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        # Expected columns: frame_id, label, x, y, score
        # We need at least frame_id, label, x, y for plotting.
        expected_min_cols = 4 

        for row_idx, row in enumerate(csv_reader):
            if len(row) < expected_min_cols:
                print(f"Skipping row {row_idx+2}: insufficient columns (need at least {expected_min_cols}). Row: {row}")
                continue
            try:
                frame_id = int(row[0])
                keypoint_label = row[1]
                x = float(row[2])
                y = float(row[3])
                # score = float(row[4]) # if score is present and needed

                if keypoint_label not in keypoint_map:
                    # Silently skip unknown keypoints or print a warning once
                    # print(f"Warning: Unknown keypoint label '{keypoint_label}' found.")
                    continue

                frames_data[frame_id].append({'label': keypoint_label, 'x': x, 'y': y})

                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)

            except ValueError as e:
                print(f"Skipping row {row_idx+2} due to ValueError: {e}. Row: {row}")
                continue
            except IndexError as e:
                print(f"Skipping row {row_idx+2} due to IndexError (likely missing x or y): {e}. Row: {row}")
                continue
    
    if not frames_data:
        print("Error: No valid keypoint data found in CSV file after processing.")
        return

    processed_keypoints_data = []
    sorted_frame_ids = sorted(frames_data.keys())

    for frame_id in sorted_frame_ids:
        keypoints_for_frame = frames_data[frame_id]
        frame_coords = [np.nan] * (num_coco_keypoints * 2) # Initialize with NaNs
        
        num_found_keypoints_in_frame = 0
        for kp_data in keypoints_for_frame:
            label = kp_data['label']
            if label in keypoint_map: # Should always be true due to earlier check
                idx = keypoint_map[label]
                frame_coords[idx * 2] = kp_data['x']
                frame_coords[idx * 2 + 1] = kp_data['y']
                num_found_keypoints_in_frame += 1
        
        if num_found_keypoints_in_frame > 0:
            processed_keypoints_data.append(frame_coords)
        # else:
            # print(f"Frame {frame_id} had no recognized keypoints, skipping.")

    if not processed_keypoints_data:
        print("Error: No frames with recognized keypoints to animate.")
        return

    final_keypoints_data = np.array(processed_keypoints_data)
    num_frames_to_animate = final_keypoints_data.shape[0]

    # Standard COCO connections (0-indexed based on keypoint_order)
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head: Nose to L/R Eye, L/R Eye to L/R Ear
        (5, 6),                          # Torso: L Shoulder to R Shoulder
        (5, 7), (7, 9),                  # Left Arm: L Shoulder to L Elbow, L Elbow to L Wrist
        (6, 8), (8, 10),                 # Right Arm: R Shoulder to R Elbow, R Elbow to R Wrist
        (5, 11), (6, 12), (11, 12),      # Torso: L/R Shoulder to L/R Hip, L Hip to R Hip
        (11, 13), (13, 15),              # Left Leg: L Hip to L Knee, L Knee to L Ankle
        (12, 14), (14, 16)               # Right Leg: R Hip to R Knee, R Knee to R Ankle
    ]

    fig, ax = plt.subplots()
    
    if num_frames_to_animate > 0 and min_x != float('inf'): # Check if any data was actually processed
        x_range = max_x - min_x
        y_range = max_y - min_y
        padding_x = x_range * 0.1 if x_range > 0 else 1.0
        padding_y = y_range * 0.1 if y_range > 0 else 1.0
        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(max_y + padding_y, min_y - padding_y) # Invert y-axis
    else: # Default limits if no data
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1) # Invert y-axis for default
        
    ax.set_aspect('equal', adjustable='box')

    plot_lines = [ax.plot([], [], '-', lw=2)[0] for _ in skeleton_connections] # Default color cycle
    plot_points = ax.plot([], [], 'o', ms=3)[0]

    def init():
        for line in plot_lines:
            line.set_data([], [])
        plot_points.set_data([], [])
        return plot_lines + [plot_points]

    def animate_frame(frame_idx):
        frame_coords = final_keypoints_data[frame_idx]
        
        all_xs = frame_coords[0::2]
        all_ys = frame_coords[1::2]

        for i, (start_idx, end_idx) in enumerate(skeleton_connections):
            x_start, y_start = frame_coords[start_idx * 2], frame_coords[start_idx * 2 + 1]
            x_end, y_end = frame_coords[end_idx * 2], frame_coords[end_idx * 2 + 1]

            if not (np.isnan(x_start) or np.isnan(y_start) or np.isnan(x_end) or np.isnan(y_end)):
                plot_lines[i].set_data([x_start, x_end], [y_start, y_end])
            else:
                plot_lines[i].set_data([], []) 

        valid_indices = ~np.isnan(all_xs) # Assuming if x is NaN, y is also NaN
        plot_points.set_data(all_xs[valid_indices], all_ys[valid_indices])
        
        return plot_lines + [plot_points]

    ani = animation.FuncAnimation(fig, animate_frame, frames=num_frames_to_animate, 
                                  init_func=init, blit=True, interval=100)

    try:
        # Attempt to save the animation as a GIF
        animation_output_path = 'skeleton_animation.gif'
        print(f"Attempting to save animation to {animation_output_path}...")
        ani.save(animation_output_path, writer='pillow', fps=10)
        print(f"Animation saved to {animation_output_path}")
        plt.close(fig) # Close the figure to free memory
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Attempting to show plot as fallback (might not work in non-interactive backend)...")
        try:
            plt.show()
        except UserWarning as uw:
            print(f"Matplotlib UserWarning during plt.show(): {uw}")
            print("This is likely due to a non-interactive backend.")
        except Exception as show_e:
            print(f"Error during plt.show(): {show_e}")


if __name__ == "__main__":
    csv_file_path = 'No_Fall/Keypoints_CSV/S_N_314_keypoints.csv'
    
    # Ensure directory for CSV exists, especially if we create a dummy file
    csv_dir = os.path.dirname(csv_file_path)
    if csv_dir and not os.path.exists(csv_dir):
        try:
            os.makedirs(csv_dir, exist_ok=True)
            print(f"Created directory: {csv_dir}")
        except Exception as e:
            print(f"Error creating directory {csv_dir}: {e}")
            # Not returning, will try to proceed; os.path.exists will catch file not found

    if not os.path.exists(csv_file_path):
        print(f"Warning: CSV file not found at {csv_file_path}")
        print(f"Creating a dummy CSV file at {csv_file_path} for testing.")
        dummy_header = "frame_id,label,x,y,score\n"
        dummy_data_lines = [
            "1,Nose,100,100,0.9", "1,Left Eye,90,90,0.9", "1,Right Eye,110,90,0.9",
            "1,Left Ear,80,80,0.9", "1,Right Ear,120,80,0.9",
            "1,Left Shoulder,70,150,0.9", "1,Right Shoulder,130,150,0.9",
            "1,Left Elbow,60,200,0.9", "1,Right Elbow,140,200,0.9",
            "1,Left Wrist,50,250,0.9", "1,Right Wrist,150,250,0.9",
            "1,Left Hip,80,250,0.9", "1,Right Hip,120,250,0.9",
            "1,Left Knee,70,300,0.9", "1,Right Knee,130,300,0.9",
            "1,Left Ankle,60,350,0.9", "1,Right Ankle,140,350,0.9",
            "2,Nose,105,105,0.9", "2,Left Eye,95,95,0.9", "2,Right Eye,115,95,0.9",
            "2,Left Ear,85,85,0.9", "2,Right Ear,125,85,0.9",
            "2,Left Shoulder,75,155,0.9", "2,Right Shoulder,135,155,0.9",
            "2,Left Elbow,65,205,0.9", "2,Right Elbow,145,205,0.9",
            "2,Left Wrist,55,255,0.9", "2,Right Wrist,155,255,0.9",
            "2,Left Hip,85,255,0.9", "2,Right Hip,125,255,0.9",
            "2,Left Knee,75,305,0.9", "2,Right Knee,135,305,0.9",
            "2,Left Ankle,65,355,0.9", "2,Right Ankle,145,355,0.9",
        ]
        try:
            with open(csv_file_path, 'w', newline='') as f:
                f.write(dummy_header)
                for line in dummy_data_lines:
                    f.write(line + "\n")
            print(f"Dummy CSV file created. Please replace it with your actual data for correct visualization.")
            display_skeleton_from_csv(csv_file_path)
        except IOError as e:
            print(f"Error creating dummy CSV file: {e}")
    else:
        display_skeleton_from_csv(csv_file_path)
