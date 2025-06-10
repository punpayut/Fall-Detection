# --- START OF FILE display_seq.py (เวอร์ชันที่ใช้ load_and_remap_sequence) ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from pathlib import Path
import traceback

# Import from common_keypoints
try:
    from common_keypoints import (
        KEYPOINT_ORDER_COCO,
        KEYPOINT_ORDER_ALPHABETICAL, # Needed for interpreting .npy
        NUM_KEYPOINTS,
        SKELETON_CONNECTIONS_INDEXED # Based on COCO order for plotting
    )
except ImportError:
    print("CRITICAL Error: common_keypoints.py not found. Please ensure it's in the same directory or Python path.")
    # Fallback definitions
    _COCO_BASE = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]
    KEYPOINT_ORDER_COCO = list(_COCO_BASE)
    KEYPOINT_ORDER_ALPHABETICAL = sorted(_COCO_BASE)
    NUM_KEYPOINTS = len(_COCO_BASE)
    SKELETON_CONNECTIONS_INDEXED = [] # Rebuild based on fallback COCO
    _SKELETON_CONNECTIONS_NAMED_FALLBACK = [
        ("Nose", "Left Eye"), ("Nose", "Right Eye"), ("Left Eye", "Left Ear"), ("Right Eye", "Right Ear"),
        ("Left Shoulder", "Right Shoulder"), ("Left Shoulder", "Left Hip"), ("Right Shoulder", "Right Hip"),
        ("Left Hip", "Right Hip"), ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
        ("Right Shoulder", "Right Elbow"), ("Right Elbow", "Right Wrist"), ("Left Hip", "Left Knee"),
        ("Left Knee", "Left Ankle"), ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"),
    ]
    for kp_name1, kp_name2 in _SKELETON_CONNECTIONS_NAMED_FALLBACK:
        try:
            idx1 = KEYPOINT_ORDER_COCO.index(kp_name1)
            idx2 = KEYPOINT_ORDER_COCO.index(kp_name2)
            SKELETON_CONNECTIONS_INDEXED.append((idx1, idx2))
        except ValueError: pass # Should not happen with _COCO_BASE


CONFIDENCE_THRESHOLD = 0.1

def load_and_remap_sequence(npy_path: Path):
    """
    Loads sequence data from a .npy file (assumed to be in KEYPOINT_ORDER_ALPHABETICAL)
    and remaps keypoints to KEYPOINT_ORDER_COCO for plotting.
    Output data structure for each frame: [X_coco1, Y_coco1, X_coco2, Y_coco2, ...]
    """
    sequence_data_alphabetical = np.load(npy_path)
    num_frames_in_seq, num_features = sequence_data_alphabetical.shape

    if num_features != NUM_KEYPOINTS * 3:
        raise ValueError(
            f"Unexpected number of features in {npy_path}. "
            f"Expected {NUM_KEYPOINTS * 3}, got {num_features}"
        )

    remapped_frames_xy_coco = np.full((num_frames_in_seq, NUM_KEYPOINTS * 2), np.nan)
    overall_min_x, overall_max_x = float('inf'), float('-inf')
    overall_min_y, overall_max_y = float('inf'), float('-inf')

    for frame_idx in range(num_frames_in_seq):
        raw_frame_data_alpha = sequence_data_alphabetical[frame_idx]
        current_frame_xy_coco_ordered = np.full(NUM_KEYPOINTS * 2, np.nan)
        
        frame_has_valid_kp = False
        current_frame_min_x, current_frame_max_x = float('inf'), float('-inf')
        current_frame_min_y, current_frame_max_y = float('inf'), float('-inf')

        for alpha_idx, kp_name_alpha in enumerate(KEYPOINT_ORDER_ALPHABETICAL):
            base_offset = alpha_idx * 3
            x = raw_frame_data_alpha[base_offset + 0]
            y = raw_frame_data_alpha[base_offset + 1]
            confidence = raw_frame_data_alpha[base_offset + 2]

            plot_x, plot_y = np.nan, np.nan 
            if confidence >= CONFIDENCE_THRESHOLD and not (np.isnan(x) or np.isnan(y)):
                plot_x, plot_y = x, y
                current_frame_min_x = min(current_frame_min_x, x)
                current_frame_max_x = max(current_frame_max_x, x)
                current_frame_min_y = min(current_frame_min_y, y)
                current_frame_max_y = max(current_frame_max_y, y)
                frame_has_valid_kp = True
            
            try:
                coco_idx = KEYPOINT_ORDER_COCO.index(kp_name_alpha)
            except ValueError:
                print(f"Warning: Keypoint '{kp_name_alpha}' not found in KEYPOINT_ORDER_COCO. Skipping.")
                continue
            
            current_frame_xy_coco_ordered[coco_idx * 2] = plot_x
            current_frame_xy_coco_ordered[coco_idx * 2 + 1] = plot_y
            
        remapped_frames_xy_coco[frame_idx] = current_frame_xy_coco_ordered
        if frame_has_valid_kp:
            overall_min_x = min(overall_min_x, current_frame_min_x)
            overall_max_x = max(overall_max_x, current_frame_max_x)
            overall_min_y = min(overall_min_y, current_frame_min_y)
            overall_max_y = max(overall_max_y, current_frame_max_y)
            
    return remapped_frames_xy_coco, (overall_min_x, overall_max_x, overall_min_y, overall_max_y)

# ... (ส่วนที่เหลือของ animate_sequence_from_npy, init_animation, update_animation_frame, if __name__ == "__main__": เหมือนเดิมที่คุณให้มาในคำตอบที่ 30/35) ...
def animate_sequence_from_npy(npy_file_path_str: str):
    npy_file_path = Path(npy_file_path_str)
    if not npy_file_path.is_file():
        print(f"Error: File not found: {npy_file_path}")
        return

    try:
        # ใช้ฟังก์ชัน load_and_remap_sequence เดิมที่คาดหวัง alphabetical input
        plot_keypoints_data, (min_x, max_x, min_y, max_y) = load_and_remap_sequence(npy_file_path)
    except Exception as e:
        print(f"Error loading or remapping sequence from {npy_file_path}: {e}")
        traceback.print_exc()
        return

    num_frames_to_animate = plot_keypoints_data.shape[0]
    if num_frames_to_animate == 0:
        print("No frames to animate after loading.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    if min_x != float('inf'):
        x_range_orig = max_x - min_x
        y_range_orig = max_y - min_y
        MIN_VISUAL_AXIS_RANGE = 150.0

        current_min_x, current_max_x = min_x, max_x
        if x_range_orig < MIN_VISUAL_AXIS_RANGE and x_range_orig > 1e-6:
            center_x = (min_x + max_x) / 2
            current_min_x = center_x - MIN_VISUAL_AXIS_RANGE / 2
            current_max_x = center_x + MIN_VISUAL_AXIS_RANGE / 2
        elif x_range_orig <= 1e-6 :
            current_min_x -= MIN_VISUAL_AXIS_RANGE / 2
            current_max_x += MIN_VISUAL_AXIS_RANGE / 2
        
        current_min_y, current_max_y = min_y, max_y
        if y_range_orig < MIN_VISUAL_AXIS_RANGE and y_range_orig > 1e-6:
            center_y = (min_y + max_y) / 2
            current_min_y = center_y - MIN_VISUAL_AXIS_RANGE / 2
            current_max_y = center_y + MIN_VISUAL_AXIS_RANGE / 2
        elif y_range_orig <= 1e-6:
            current_min_y -= MIN_VISUAL_AXIS_RANGE / 2
            current_max_y += MIN_VISUAL_AXIS_RANGE / 2
            
        final_x_range = current_max_x - current_min_x
        final_y_range = current_max_y - current_min_y

        padding_x = final_x_range * 0.10
        padding_y = final_y_range * 0.10
        
        ax.set_xlim(current_min_x - padding_x, current_max_x + padding_x)
        ax.set_ylim(current_max_y + padding_y, current_min_y - padding_y)
    else:
        print("Warning: No valid keypoints found with sufficient confidence. Using default plot limits.")
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)
        
    ax.set_aspect('equal', adjustable='box')
    plot_lines = [ax.plot([], [], '-', lw=2)[0] for _ in SKELETON_CONNECTIONS_INDEXED]
    plot_points = ax.plot([], [], 'o', ms=4, color='red')[0]

    def init_animation():
        for line in plot_lines:
            line.set_data([], [])
        plot_points.set_data([], [])
        ax.set_title(f"Skeleton: {npy_file_path.name}")
        return plot_lines + [plot_points]

    def update_animation_frame(frame_idx):
        frame_coords_xy = plot_keypoints_data[frame_idx] # This is now [X_coco1,Y_coco1,X_coco2,Y_coco2,...]
        
        all_xs = frame_coords_xy[0::2]
        all_ys = frame_coords_xy[1::2]

        for i, (start_idx_coco, end_idx_coco) in enumerate(SKELETON_CONNECTIONS_INDEXED):
            x_start = frame_coords_xy[start_idx_coco * 2]
            y_start = frame_coords_xy[start_idx_coco * 2 + 1]
            x_end = frame_coords_xy[end_idx_coco * 2]
            y_end = frame_coords_xy[end_idx_coco * 2 + 1]

            if not (np.isnan(x_start) or np.isnan(y_start) or np.isnan(x_end) or np.isnan(y_end)):
                plot_lines[i].set_data([x_start, x_end], [y_start, y_end])
            else:
                plot_lines[i].set_data([], [])

        valid_kp_mask = ~np.isnan(all_xs)
        plot_points.set_data(all_xs[valid_kp_mask], all_ys[valid_kp_mask])
        
        ax.set_title(f"Skeleton: {npy_file_path.name} - Frame {frame_idx + 1}/{num_frames_to_animate}")
        return plot_lines + [plot_points]

    ani = animation.FuncAnimation(fig, update_animation_frame, frames=num_frames_to_animate,
                                  init_func=init_animation, blit=True, interval=100)

    output_gif_path = npy_file_path.with_suffix(".gif")
    try:
        print(f"Attempting to save animation to {output_gif_path}...")
        ani.save(str(output_gif_path), writer='pillow', fps=10) 
        print(f"Animation successfully saved to {output_gif_path}")
    except Exception as e:
        print(f"Error saving animation to GIF: {e}")
        print("Ensure 'pillow' is installed. You might also need 'imagemagick' or 'ffmpeg'.")
    finally:
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display skeleton animation from a .npy sequence file and save it as a GIF."
    )
    parser.add_argument(
        "npy_file_path",
        type=str,
        help="Path to the .npy sequence file"
    )
    args = parser.parse_args()
    if not Path(args.npy_file_path).exists():
        print(f"Error: The provided .npy file path does not exist: {args.npy_file_path}")
    else:
        animate_sequence_from_npy(args.npy_file_path)
# --- END OF FILE display_seq.py ---