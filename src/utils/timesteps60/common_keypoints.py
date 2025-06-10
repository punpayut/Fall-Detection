# --- START OF FILE common_keypoints.py ---
import numpy as np

# Base list of keypoint names in COCO order
# This is the "source of truth" for keypoint names.
COCO_KEYPOINT_NAMES_BASE = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

NUM_KEYPOINTS = len(COCO_KEYPOINT_NAMES_BASE)

# Order for direct use (e.g., reading CSVs structured this way, MediaPipe output order)
# and for final display mapping.
KEYPOINT_ORDER_COCO = list(COCO_KEYPOINT_NAMES_BASE) # Explicit COCO order

# Order for .npy file structure if keypoints are stored alphabetically by some process.
# display_sequence_animation.py will use this to interpret .npy files.
KEYPOINT_ORDER_ALPHABETICAL = sorted(COCO_KEYPOINT_NAMES_BASE)

# --- Skeleton Connections ---
# Define connections using human-readable names first.
# This list should match the SKELETON_CONNECTIONS in display_sequence_animation.py initially.
# It also aligns with the connections used in display_skeleton.py (though slightly different structure)
SKELETON_CONNECTIONS_NAMED = [
    # Head
    ("Nose", "Left Eye"), ("Nose", "Right Eye"),
    ("Left Eye", "Left Ear"), ("Right Eye", "Right Ear"),
    # Torso
    ("Left Shoulder", "Right Shoulder"),
    ("Left Shoulder", "Left Hip"),
    ("Right Shoulder", "Right Hip"),
    ("Left Hip", "Right Hip"),
    # Left Arm
    ("Left Shoulder", "Left Elbow"),
    ("Left Elbow", "Left Wrist"),
    # Right Arm
    ("Right Shoulder", "Right Elbow"),
    ("Right Elbow", "Right Wrist"),
    # Left Leg
    ("Left Hip", "Left Knee"),
    ("Left Knee", "Left Ankle"),
    # Right Leg
    ("Right Hip", "Right Knee"),
    ("Right Knee", "Right Ankle"),
]

# Convert named connections to indexed connections based on KEYPOINT_ORDER_COCO
# This will be used by plotting functions.
SKELETON_CONNECTIONS_INDEXED = []
for kp_name1, kp_name2 in SKELETON_CONNECTIONS_NAMED:
    try:
        idx1 = KEYPOINT_ORDER_COCO.index(kp_name1)
        idx2 = KEYPOINT_ORDER_COCO.index(kp_name2)
        SKELETON_CONNECTIONS_INDEXED.append((idx1, idx2))
    except ValueError as e:
        print(f"ERROR in common_keypoints.py: SKELETON_CONNECTIONS_NAMED contains a keypoint name"
              f" not found in KEYPOINT_ORDER_COCO - {e}")
        # Decide how to handle: raise error, or skip this connection
        # For now, let it proceed, but this indicates a definition mismatch.

if __name__ == '__main__':
    print("--- common_keypoints.py Definitions ---")
    print(f"NUM_KEYPOINTS: {NUM_KEYPOINTS}")
    print("\nKEYPOINT_ORDER_COCO:")
    for i, name in enumerate(KEYPOINT_ORDER_COCO):
        print(f"  {i}: {name}")

    print("\nKEYPOINT_ORDER_ALPHABETICAL:")
    for i, name in enumerate(KEYPOINT_ORDER_ALPHABETICAL):
        print(f"  {i}: {name}")

    print("\nSKELETON_CONNECTIONS_INDEXED (based on KEYPOINT_ORDER_COCO indices):")
    for conn in SKELETON_CONNECTIONS_INDEXED:
        print(f"  ({conn[0]}: {KEYPOINT_ORDER_COCO[conn[0]]}, {conn[1]}: {KEYPOINT_ORDER_COCO[conn[1]]})")
# --- END OF FILE common_keypoints.py ---