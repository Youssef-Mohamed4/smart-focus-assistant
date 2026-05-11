import cv2
import mediapipe as mp
import numpy as np
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def get_path(model_name):
    return os.path.join(SCRIPT_DIR, model_name)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Heuristic Thresholds
# Press 'c' while running to dynamically calibrate posture!
global_slouch_threshold = 0.28 
GAZE_TURN_LEFT = 0.82
GAZE_TURN_RIGHT = 1.25
PITCH_LOOK_UP = 0.35
PITCH_LOOK_DOWN = 0.9

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=get_path('face_landmarker.task')),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=get_path('pose_landmarker.task')),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6)

segmenter_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=get_path('selfie_segmenter.tflite')),
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture(0)

try:
    with FaceLandmarker.create_from_options(face_options) as face_detector, \
         PoseLandmarker.create_from_options(pose_options) as pose_detector, \
         ImageSegmenter.create_from_options(segmenter_options) as segmenter:

        print("Modern System Online. Press 'ESC' or 'X' to exit.")

        # Smoothing prevents rapid flickering of data
        smoothed_p = None
        smoothed_g = None
        smoothed_pitch = None
        EMA_ALPHA = 0.15
        
        is_calibrating = False
        calibration_frames = []

        cv2.namedWindow('smart-focus-assistant', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('smart-focus-assistant', 1000, 750)
        cv2.moveWindow('smart-focus-assistant', 460, 165)

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            mask = seg_result.confidence_masks[0].numpy_view()
            mask = np.squeeze(mask)
            
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            condition = np.stack((mask,) * 3, axis=-1) > 0.4
            blurred_bg = cv2.GaussianBlur(frame, (99, 99), 0)
            output_image = np.where(condition, frame, blurred_bg)

            pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
            posture_status = "Good Posture"
            p_color = (0, 255, 0)
            live_p = 0.0

            if pose_result.pose_landmarks:
                marks = pose_result.pose_landmarks[0]
                ear_y = (marks[7].y + marks[8].y) / 2
                shld_y = (marks[11].y + marks[12].y) / 2
                
                raw_p = shld_y - ear_y
                
                if smoothed_p is None: smoothed_p = raw_p
                else: smoothed_p = (EMA_ALPHA * raw_p) + ((1 - EMA_ALPHA) * smoothed_p)
                live_p = smoothed_p
                
                if is_calibrating:
                    calibration_frames.append(live_p)
                    posture_status = f"Calibrating... {len(calibration_frames)}/30"
                    p_color = (255, 255, 0)
                    if len(calibration_frames) >= 30:
                        is_calibrating = False
                        avg_p = sum(calibration_frames) / len(calibration_frames)
                        global_slouch_threshold = avg_p * 0.90
                        calibration_frames = []
                        print(f"Calibration Complete! New Slouch Threshold: {global_slouch_threshold:.3f}")
                else:
                    if live_p < global_slouch_threshold:
                        posture_status = "Slouching Warning!"
                        p_color = (0, 0, 255)
            else:
                posture_status = "No Person Detected"
                p_color = (128, 128, 128)
                smoothed_p = None

            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
            focus_status = "Focused"
            f_color = (0, 255, 0)
            live_yaw = 1.0
            live_pitch = 0.6

            if face_result.face_landmarks:
                face = face_result.face_landmarks[0]
                
                # Key landmarks: Nose tip (1), Chin (152), Left Eye (33), Right Eye (263)
                nose = face[1]
                chin = face[152]
                left_eye = face[33]
                right_eye = face[263]
                
                dist_left = np.sqrt((nose.x - left_eye.x)**2 + (nose.y - left_eye.y)**2)
                dist_right = np.sqrt((nose.x - right_eye.x)**2 + (nose.y - right_eye.y)**2)
                
                raw_yaw = dist_left / dist_right if dist_right > 0 else 1.0
                
                if smoothed_g is None: smoothed_g = raw_yaw
                else: smoothed_g = (EMA_ALPHA * raw_yaw) + ((1 - EMA_ALPHA) * smoothed_g)
                live_yaw = smoothed_g
                
                eye_midpoint_y = (left_eye.y + right_eye.y) / 2
                dist_nose_to_eyes = nose.y - eye_midpoint_y
                dist_nose_to_chin = chin.y - nose.y
                
                raw_pitch = dist_nose_to_eyes / dist_nose_to_chin if dist_nose_to_chin > 0 else 0.6
                
                if smoothed_pitch is None: smoothed_pitch = raw_pitch
                else: smoothed_pitch = (EMA_ALPHA * raw_pitch) + ((1 - EMA_ALPHA) * smoothed_pitch)
                live_pitch = smoothed_pitch
                
                if live_yaw < GAZE_TURN_LEFT or live_yaw > GAZE_TURN_RIGHT:
                    focus_status = "Looking Away (Side)"
                    f_color = (0, 0, 255)
                elif live_pitch < PITCH_LOOK_UP or live_pitch > PITCH_LOOK_DOWN:
                    focus_status = "Looking Away (Up/Down)"
                    f_color = (0, 0, 255)
            else:
                focus_status = "No Face Detected"
                f_color = (128, 128, 128)
                smoothed_g = None
                smoothed_pitch = None

            overlay = output_image.copy()
            cv2.rectangle(overlay, (10, 10), (380, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, output_image, 0.4, 0, output_image)
            
            cv2.putText(output_image, f"Posture: {posture_status}", (25, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, p_color, 1)
            cv2.putText(output_image, f"Attention: {focus_status}", (25, 70), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, f_color, 1)
            cv2.putText(output_image, f"P: {live_p:.2f} / {global_slouch_threshold:.2f} | Y: {live_yaw:.2f} | Pi: {live_pitch:.2f}", (25, 110), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow('smart-focus-assistant', output_image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('x'): break
            if key == ord('c') and not is_calibrating:
                is_calibrating = True
                calibration_frames = []
                
            try:
                if cv2.getWindowProperty('smart-focus-assistant', cv2.WND_PROP_VISIBLE) < 1: break
            except cv2.error:
                break

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()