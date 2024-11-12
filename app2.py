import cv2
import numpy as np
import time
import threading
import os
import joblib
import pywhatkit as kit
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Face Detection and Pose Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(1)

# Load pre-trained model for fainting detection
model = joblib.load('modelok.pkl')

# Variables for heart rate calculation
prev_time = time.time()
display_time = time.time()
heart_rate = 0
displayed_heart_rate = 0
frame_count = 0
pulse_buffer = []
alpha = 0.1  # Smoothing parameter for low-pass filter

# "Post-Exercise" simulation trigger
exercise_start = False
exercise_trigger_frame = 300  # Trigger after 300 frames (~10 seconds)

# WhatsApp Alert Configuration
whatsapp_number = "+917204677562"

# Function to calculate heart rate
def calculate_heart_rate(pulse_buffer):
    return np.mean(pulse_buffer) if pulse_buffer else 0

# Normalize function for heart rate
def normalize_value(value, min_val=60, max_val=100):
    if pulse_buffer:
        min_pulse, max_pulse = min(pulse_buffer), max(pulse_buffer)
        scaled_value = (value - min_pulse) / (max_pulse - min_pulse) if max_pulse > min_pulse else 0
        normalized_value = min_val + scaled_value * (max_val - min_val)

        # Adjust range during "exercise" or if fainting
        if exercise_start:
            normalized_value = min(90, max(80, normalized_value))
        elif faint_detected:
            normalized_value = max(50, min(60, normalized_value))

        return min(normalized_value, max_val)
    return min_val

# Function to send WhatsApp alert
def send_whatsapp_alert(current_heart_rate, image_path=None):
    alert_message = f"Fainting detected! Last recorded heart rate: {current_heart_rate:.2f} BPM. Please check immediately."
    
    def alert():
        try:
            kit.sendwhatmsg_instantly(whatsapp_number, alert_message)
            print("WhatsApp message sent successfully!")
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")

    threading.Thread(target=alert).start()



# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(rgb_frame)
    results_pose = pose.process(rgb_frame)

    # Heart rate estimation
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Calculate average color intensity in face region
            face_region = frame[y:y+h, x:x+w]
            avg_color = np.mean(face_region, axis=(0, 1))
            pulse_value = np.linalg.norm(avg_color)

            # Apply a low-pass filter
            if pulse_buffer:
                pulse_value = alpha * pulse_value + (1 - alpha) * pulse_buffer[-1]
            pulse_buffer.append(pulse_value)

            # Maintain only the last 30 frames
            if len(pulse_buffer) > 30:
                pulse_buffer.pop(0)

            # Heart rate update every 2 seconds
            current_time = time.time()
            if current_time - display_time >= 2:
                display_time = current_time
                heart_rate = calculate_heart_rate(pulse_buffer)
                displayed_heart_rate = normalize_value(heart_rate)

            # Display the current heart rate
            cv2.putText(frame, f'Heart Rate: {displayed_heart_rate:.2f} BPM', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Pose detection for fainting, sitting, standing
    faint_detected = False
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        points = [
            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
            (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
            (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
        ]
        feature_vector = np.array(points).flatten().reshape(1, -1)
        prediction = model.predict(feature_vector)
        prediction_prob = model.predict_proba(feature_vector)

        prob_faint = prediction_prob[0][1] * 100
        prob_sitting = prediction_prob[0][2] * 100
        prob_standing = prediction_prob[0][3] * 100


        faint_detected = prob_faint > 80
        

        # Display probabilities for each detected pose
        cv2.putText(frame, f'Faint Probability: {prob_faint:.2f}%', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'Sitting Probability: {prob_sitting:.2f}%', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Standing Probability: {prob_standing:.2f}%', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Highlight current detected pose
        if faint_detected:
            cv2.putText(frame, 'Faint Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            send_whatsapp_alert(displayed_heart_rate, frame)
            break
        elif prob_sitting > 90:
            cv2.putText(frame, 'Sitting Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif prob_standing > 7:
            cv2.putText(frame, 'Standing Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw pose landmarks
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1
                                             ])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Heart Rate and Pose Detection", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
