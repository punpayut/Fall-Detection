import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from collections import deque
import gradio as gr
import os
import shutil # สำหรับ copy ไฟล์ example

# --- Configuration (ปรับให้เหมาะกับ Gradio) ---
MODEL_PATH = 'fall_detection_transformer.tflite'
INPUT_TIMESTEPS = 30
FALL_CONFIDENCE_THRESHOLD = 0.90
MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION = 0.3
mp_pose = mp.solutions.pose
pose_complexity = 0 # ลด complexity เพื่อความเร็วบน Spaces, ลอง 0 หรือ 1
use_static_image_mode = False # สำหรับวิดีโอไฟล์ จะถูก override เป็น True ใน process_video

FALL_EVENT_COOLDOWN = 10

# ----- 0. KEYPOINT DEFINITIONS (เหมือนเดิม) -----
KEYPOINT_NAMES_ORIGINAL = [
    'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', 'Right Eye', 'Right Eye Outer',
    'Left Ear', 'Right Ear', 'Mouth Left', 'Mouth Right',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist',
    'Left Pinky', 'Right Pinky', 'Left Index', 'Right Index', 'Left Thumb', 'Right Thumb',
    'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
    'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index'
]
MEDIAPIPE_TO_YOUR_KEYPOINTS_MAPPING = {
    mp_pose.PoseLandmark.NOSE: 'Nose', mp_pose.PoseLandmark.LEFT_EYE: 'Left Eye',
    mp_pose.PoseLandmark.RIGHT_EYE: 'Right Eye', mp_pose.PoseLandmark.LEFT_EAR: 'Left Ear',
    mp_pose.PoseLandmark.RIGHT_EAR: 'Right Ear', mp_pose.PoseLandmark.LEFT_SHOULDER: 'Left Shoulder',
    mp_pose.PoseLandmark.RIGHT_SHOULDER: 'Right Shoulder', mp_pose.PoseLandmark.LEFT_ELBOW: 'Left Elbow',
    mp_pose.PoseLandmark.RIGHT_ELBOW: 'Right Elbow', mp_pose.PoseLandmark.LEFT_WRIST: 'Left Wrist',
    mp_pose.PoseLandmark.RIGHT_WRIST: 'Right Wrist', mp_pose.PoseLandmark.LEFT_HIP: 'Left Hip',
    mp_pose.PoseLandmark.RIGHT_HIP: 'Right Hip', mp_pose.PoseLandmark.LEFT_KNEE: 'Left Knee',
    mp_pose.PoseLandmark.RIGHT_KNEE: 'Right Knee', mp_pose.PoseLandmark.LEFT_ANKLE: 'Left Ankle',
    mp_pose.PoseLandmark.RIGHT_ANKLE: 'Right Ankle'
}
YOUR_KEYPOINT_NAMES_TRAINING = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]
SORTED_YOUR_KEYPOINT_NAMES = sorted(YOUR_KEYPOINT_NAMES_TRAINING)
KEYPOINT_DICT_TRAINING = {name: i for i, name in enumerate(SORTED_YOUR_KEYPOINT_NAMES)}
NUM_KEYPOINTS_TRAINING = len(KEYPOINT_DICT_TRAINING)
NUM_FEATURES = NUM_KEYPOINTS_TRAINING * 3

print("--- Initializing Keypoint Definitions for Gradio App ---")
print(f"NUM_FEATURES for model input: {NUM_FEATURES}")
# ---------------------------------------------------------------

# --- Load TFLite Model ---
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite Model Loaded: {MODEL_PATH}")
    model_expected_shape = tuple(input_details[0]['shape'])
    if model_expected_shape[2] != NUM_FEATURES or model_expected_shape[1] != INPUT_TIMESTEPS:
        print(f"FATAL ERROR: Model's expected input shape features/timesteps "
              f"({model_expected_shape[1]},{model_expected_shape[2]}) "
              f"does not match configured ({INPUT_TIMESTEPS},{NUM_FEATURES}).")
        exit()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()

# --- Helper Functions (get_kpt_indices, normalize_skeleton_frame, extract_and_normalize_features - เหมือนเดิม) ---
def get_kpt_indices_training_order(keypoint_name):
    if keypoint_name not in KEYPOINT_DICT_TRAINING:
        raise ValueError(f"Keypoint '{keypoint_name}' not found in KEYPOINT_DICT_TRAINING. Available: {list(KEYPOINT_DICT_TRAINING.keys())}")
    kp_idx = KEYPOINT_DICT_TRAINING[keypoint_name]
    return kp_idx * 3, kp_idx * 3 + 1, kp_idx * 3 + 2

def normalize_skeleton_frame(frame_features_sorted, min_confidence=MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION):
    normalized_frame = np.copy(frame_features_sorted)
    ref_kp_names = {'ls': 'Left Shoulder', 'rs': 'Right Shoulder', 'lh': 'Left Hip', 'rh': 'Right Hip'}
    try:
        ls_x_idx, ls_y_idx, ls_c_idx = get_kpt_indices_training_order(ref_kp_names['ls'])
        rs_x_idx, rs_y_idx, rs_c_idx = get_kpt_indices_training_order(ref_kp_names['rs'])
        lh_x_idx, lh_y_idx, lh_c_idx = get_kpt_indices_training_order(ref_kp_names['lh'])
        rh_x_idx, rh_y_idx, rh_c_idx = get_kpt_indices_training_order(ref_kp_names['rh'])
    except ValueError as e:
        print(f"Warning in normalize_skeleton_frame (get_kpt_indices): {e}")
        return frame_features_sorted

    ls_x, ls_y, ls_c = frame_features_sorted[ls_x_idx], frame_features_sorted[ls_y_idx], frame_features_sorted[ls_c_idx]
    rs_x, rs_y, rs_c = frame_features_sorted[rs_x_idx], frame_features_sorted[rs_y_idx], frame_features_sorted[rs_c_idx]
    lh_x, lh_y, lh_c = frame_features_sorted[lh_x_idx], frame_features_sorted[lh_y_idx], frame_features_sorted[lh_c_idx]
    rh_x, rh_y, rh_c = frame_features_sorted[rh_x_idx], frame_features_sorted[rh_y_idx], frame_features_sorted[rh_c_idx]

    mid_shoulder_x, mid_shoulder_y = np.nan, np.nan
    valid_ls, valid_rs = ls_c > min_confidence, rs_c > min_confidence
    if valid_ls and valid_rs: mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    elif valid_ls: mid_shoulder_x, mid_shoulder_y = ls_x, ls_y
    elif valid_rs: mid_shoulder_x, mid_shoulder_y = rs_x, rs_y

    mid_hip_x, mid_hip_y = np.nan, np.nan
    valid_lh, valid_rh = lh_c > min_confidence, rh_c > min_confidence
    if valid_lh and valid_rh: mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
    elif valid_lh: mid_hip_x, mid_hip_y = lh_x, lh_y
    elif valid_rh: mid_hip_x, mid_hip_y = rh_x, rh_y

    if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
        return frame_features_sorted

    reference_height = np.nan
    if not np.isnan(mid_shoulder_y) and not np.isnan(mid_hip_y):
        reference_height = np.abs(mid_shoulder_y - mid_hip_y)

    perform_scaling = not (np.isnan(reference_height) or reference_height < 1e-5)

    for kp_name_sorted in SORTED_YOUR_KEYPOINT_NAMES:
        try:
            x_col, y_col, _ = get_kpt_indices_training_order(kp_name_sorted)
            normalized_frame[x_col] -= mid_hip_x
            normalized_frame[y_col] -= mid_hip_y
            if perform_scaling:
                normalized_frame[x_col] /= reference_height
                normalized_frame[y_col] /= reference_height
        except ValueError: # Should not happen if kp_name_sorted is from SORTED_YOUR_KEYPOINT_NAMES
            pass
    return normalized_frame

def extract_and_normalize_features(pose_results):
    frame_features_sorted = np.zeros(NUM_FEATURES, dtype=np.float32)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        for mp_landmark_enum, your_kp_name in MEDIAPIPE_TO_YOUR_KEYPOINTS_MAPPING.items():
            if your_kp_name in KEYPOINT_DICT_TRAINING:
                try:
                    lm = landmarks[mp_landmark_enum.value]
                    x_idx, y_idx, c_idx = get_kpt_indices_training_order(your_kp_name)
                    frame_features_sorted[x_idx], frame_features_sorted[y_idx], frame_features_sorted[c_idx] = lm.x, lm.y, lm.visibility
                except (IndexError, ValueError) as e:
                    print(f"Warning in extract_and_normalize_features for {your_kp_name}: {e}")
                    pass
    normalized_features = normalize_skeleton_frame(frame_features_sorted.copy())
    return normalized_features
# -------------------------------------------------------------------------------------------------------------------

# --- Function to process uploaded video for Gradio ---
def process_video_for_gradio(uploaded_video_path_temp):
    if uploaded_video_path_temp is None:
        return None, "Please upload a video file."

    print(f"Gradio provided temp video path: {uploaded_video_path_temp}")
    base_name = os.path.basename(uploaded_video_path_temp)
    # สร้าง path ที่ unique มากขึ้นสำหรับไฟล์ที่ copy มา
    timestamp_str = str(int(time.time() * 1000)) # เพิ่ม timestamp เพื่อความ unique
    local_video_path = os.path.join(os.getcwd(), f"{timestamp_str}_{base_name}") 

    try:
        print(f"Copying video from {uploaded_video_path_temp} to {local_video_path}")
        shutil.copy2(uploaded_video_path_temp, local_video_path)
        print(f"Video copied successfully to {local_video_path}")
    except Exception as e:
        error_msg = f"Error copying video file: {e}\nTemp path: {uploaded_video_path_temp}"
        print(error_msg); return None, error_msg

    local_feature_sequence = deque(maxlen=INPUT_TIMESTEPS)
    local_last_fall_event_time = 0 # ใช้ local_last_fall_event_time_sec เพื่อความชัดเจนว่าเป็นหน่วยวินาทีของวิดีโอ
    
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        error_msg = f"Error: OpenCV cannot open video file at copied path: {local_video_path}"
        if os.path.exists(local_video_path): print(f"File size of '{local_video_path}': {os.path.getsize(local_video_path)} bytes")
        else: print(f"File '{local_video_path}' does not exist after copy attempt.")
        if os.path.exists(local_video_path): os.remove(local_video_path) # Cleanup
        return None, error_msg

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps) or fps < 1: fps = 25.0 # Default FPS, ensure it's float
    
    processed_frames_list = []
    overall_status_updates = []

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=pose_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        
        frame_count = 0
        while cap.isOpened():
            success, original_bgr_frame = cap.read() # อ่าน frame มาเป็น BGR
            if not success:
                break
            
            frame_count += 1

            # *** START: การแก้ไขเรื่องสีและการวาด ***
            # สร้างสำเนาของ BGR frame สำหรับการวาดผลลัพธ์
            frame_for_display = original_bgr_frame.copy()

            # 1. แปลงเป็น RGB เฉพาะตอนส่งให้ MediaPipe
            image_rgb_for_mediapipe = cv2.cvtColor(original_bgr_frame, cv2.COLOR_BGR2RGB)
            image_rgb_for_mediapipe.flags.writeable = False
            results = pose.process(image_rgb_for_mediapipe)
            # image_rgb_for_mediapipe.flags.writeable = True # ไม่จำเป็นแล้ว

            # 2. Extract and Normalize Features
            current_features = extract_and_normalize_features(results)
            local_feature_sequence.append(current_features)
            
            # ... (ส่วนการทำนายผล prediction เหมือนเดิม) ...
            current_status_text_for_log = f"Frame {frame_count}: Collecting..." # สำหรับ log
            prediction_label = "no_fall"
            display_confidence_value = 0.0

            if len(local_feature_sequence) == INPUT_TIMESTEPS:
                model_input_data = np.array(local_feature_sequence, dtype=np.float32)
                model_input_data = np.expand_dims(model_input_data, axis=0)
                try:
                    interpreter.set_tensor(input_details[0]['index'], model_input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    prediction_probability_fall = output_data[0][0]

                    if prediction_probability_fall >= FALL_CONFIDENCE_THRESHOLD:
                        prediction_label = "fall"
                        display_confidence_value = prediction_probability_fall
                    else:
                        prediction_label = "no_fall"
                        display_confidence_value = 1.0 - prediction_probability_fall
                    
                    current_status_text_for_log = f"Frame {frame_count}: {prediction_label.upper()} (Conf: {display_confidence_value:.2f})"

                    current_video_time_sec = frame_count / fps
                    if prediction_label == "fall":
                        if (current_video_time_sec - local_last_fall_event_time) > FALL_EVENT_COOLDOWN: # ใช้ local_last_fall_event_time
                            fall_message = f"Frame {frame_count} (~{current_video_time_sec:.1f}s): FALL DETECTED! (Conf: {prediction_probability_fall:.2f})"
                            print(fall_message)
                            overall_status_updates.append(fall_message)
                            local_last_fall_event_time = current_video_time_sec # อัปเดตเวลา
                except Exception as e:
                    print(f"Frame {frame_count}: Error during prediction: {e}")
                    current_status_text_for_log = f"Frame {frame_count}: Prediction Error"
                    display_confidence_value = 0.0
            
            # อัปเดต overall_status_updates โดยใช้ current_status_text_for_log
            if "FALL DETECTED" not in current_status_text_for_log and \
               (frame_count % int(fps*1) == 0 or (len(local_feature_sequence) == INPUT_TIMESTEPS and frame_count == INPUT_TIMESTEPS) or frame_count ==1) :
                 if "Collecting..." not in current_status_text_for_log or frame_count == 1 :
                    overall_status_updates.append(current_status_text_for_log)


            # 3. วาด Landmarks (ถ้ามี) บน frame_for_display (BGR)
            if results.pose_landmarks:
                # เพื่อให้ได้สี default ของ MediaPipe ที่ถูกต้องที่สุด, เราจะวาดบนสำเนา RGB ชั่วคราว
                # แล้วค่อยแปลงกลับมาเป็น BGR เพื่อใส่ใน frame_for_display
                temp_rgb_to_draw_landmarks = cv2.cvtColor(original_bgr_frame, cv2.COLOR_BGR2RGB).copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    temp_rgb_to_draw_landmarks,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                # ตอนนี้ frame_for_display ยังเป็น BGR ดั้งเดิม, เราจะเอา temp_rgb_to_draw_landmarks ที่วาดแล้ว
                # แปลงกลับเป็น BGR แล้วใช้เป็น frame_for_display ใหม่
                frame_for_display = cv2.cvtColor(temp_rgb_to_draw_landmarks, cv2.COLOR_RGB2BGR)
            # ถ้าไม่มี landmarks, frame_for_display จะยังคงเป็น original_bgr_frame.copy()

            # 4. วาด Text บน frame_for_display (BGR) ทางขวามือ
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale_status = 0.6
            thickness_status = 1
            font_scale_alert = 1
            thickness_alert = 2
            padding = 30 # ระยะห่างจากขอบ

            text_to_show_on_frame = f"{prediction_label.upper()} (Conf: {display_confidence_value:.2f})"
            if "Collecting" in current_status_text_for_log or "Error" in current_status_text_for_log: # ใช้ current_status_text_for_log
                 text_to_show_on_frame = current_status_text_for_log.split(': ')[-1]

            (text_w, text_h), _ = cv2.getTextSize(text_to_show_on_frame, font_face, font_scale_status, thickness_status)
            text_x_status = frame_for_display.shape[1] - text_w - padding
            text_y_status = padding + text_h

            status_color_bgr = (255, 255, 255) # เขียว (BGR)
            current_video_time_sec_for_alert_check = frame_count / fps
            if prediction_label == "fall" and not (current_video_time_sec_for_alert_check - local_last_fall_event_time < FALL_EVENT_COOLDOWN):
                status_color_bgr = (0, 165, 255) # สีส้ม (BGR)
            if "Error" in text_to_show_on_frame:
                status_color_bgr = (0,0,255) # สีแดง (BGR)

            cv2.putText(frame_for_display, text_to_show_on_frame, (text_x_status, text_y_status), font_face, font_scale_status, status_color_bgr, thickness_status, cv2.LINE_AA)
            
            if prediction_label == "fall" and (current_video_time_sec_for_alert_check - local_last_fall_event_time < FALL_EVENT_COOLDOWN):
                alert_text = "FALL DETECTED!"
                (alert_w, alert_h), _ = cv2.getTextSize(alert_text, font_face, font_scale_alert, thickness_alert)
                alert_x_pos = frame_for_display.shape[1] - alert_w - padding
                alert_y_pos = text_y_status + alert_h + padding // 2
                cv2.putText(frame_for_display, alert_text, (alert_x_pos, alert_y_pos), font_face, font_scale_alert, (0, 0, 255), thickness_alert, cv2.LINE_AA) # สีแดง (BGR)
            
            # *** END ***
            processed_frames_list.append(frame_for_display) # เพิ่ม BGR frame ที่วาดแล้ว

    cap.release()

    if not processed_frames_list:
        if os.path.exists(local_video_path):
            try: os.remove(local_video_path); print(f"Cleaned up temp copied file: {local_video_path}")
            except Exception as e: print(f"Could not remove temp copied file {local_video_path} after no frames: {e}")
        return None, "No frames processed. Video might be empty or unreadable after copy."

    output_temp_video_path = f"processed_gradio_output_{timestamp_str}.mp4"
    height, width, _ = processed_frames_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_temp_video_path, fourcc, fps, (width, height))
    for frame_out_bgr in processed_frames_list:
        video_writer.write(frame_out_bgr)
    video_writer.release()
    print(f"Processed video saved to: {output_temp_video_path}")
    
    summary_text = "Recent Events / Status:\n" + "\n".join(overall_status_updates[-15:])

    if os.path.exists(local_video_path):
        try: os.remove(local_video_path); print(f"Cleaned up temp copied file: {local_video_path}")
        except Exception as e: print(f"Could not remove temp copied file {local_video_path}: {e}")

    return output_temp_video_path, summary_text


# --- สร้าง Gradio Interface ---

# กำหนด list ของชื่อไฟล์ตัวอย่างของคุณ
example_filenames = [
    "fall_example_1.mp4",     # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
    "fall_example_2.mp4",     # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
    "fall_example_3.mp4",  # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
    "fall_example_4.mp4"   # <<<< แก้ไขชื่อไฟล์ตามที่คุณใช้
]

examples_list_for_gradio = []
for filename in example_filenames:
    # ตรวจสอบว่าไฟล์ example มีอยู่ใน root directory ของ repo จริงๆ
    if os.path.exists(filename): # Gradio examples ต้องการแค่ชื่อไฟล์ (ถ้าอยู่ใน root)
        examples_list_for_gradio.append([filename]) # Gradio ต้องการ list ของ list
        print(f"Info: Example file '{filename}' found and added.")
    else:
        print(f"Warning: Example file '{filename}' not found in the repository root. It will not be added to examples.")

iface = gr.Interface(
    fn=process_video_for_gradio,
    inputs=gr.Video(label="Upload Video File (.mp4)", sources=["upload"]),
    outputs=[
        gr.Video(label="Processed Video with Detections"),
        gr.Textbox(label="Detection Summary (Events / Status)")
    ],
    title="AI Fall Detection from Video",
    description="Upload a video file (MP4 format recommended) to detect falls. " \
                "Processing may take time depending on video length.",
    examples=examples_list_for_gradio if examples_list_for_gradio else None, # <<<< ใช้ list ใหม่นี้
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    iface.launch()