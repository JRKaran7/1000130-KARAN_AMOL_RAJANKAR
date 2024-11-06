import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Initialize MediaPipe Pose class 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angles or extract landmarks for each frame
def extract_features_from_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        # Convert image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract relevant points (15 points in this example)
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
            
            # Flatten the list of points into a single vector (use for machine learning model)
            feature_vector = np.array(points).flatten()
            features.append(feature_vector)
        
        frame_count += 1
    
    cap.release()
    return features

# Loop over all videos and extract features
def extract_dataset_features(video_dir, label, max_frames_per_video=None):
    dataset = []
    for video in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video)
        features = extract_features_from_video(video_path, max_frames=max_frames_per_video)
        dataset.extend([(feature, label) for feature in features])
    return dataset

# Extract features from different classes of videos
faint_videos = extract_dataset_features('faint', 1)
not_faint_videos = extract_dataset_features('not_faint', 0)
sitting_videos = extract_dataset_features('sitting', 2)
standing_videos = extract_dataset_features('standing', 3)

# Combine all features and labels
dataset = faint_videos + not_faint_videos + sitting_videos + standing_videos

# Separate features and labels
X = [item[0] for item in dataset]  # Features (landmarks/angles)
y = [item[1] for item in dataset]  # Labels (1 for faint, 0 for not faint, 2 for sitting, 3 for standing)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3.1.1: Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 3.1.2: Evaluate the model
y_pred = clf.predict(X_test)
acc1 = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc1}')

# Step 3.2.1: Train a KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Step 3.2.2: Evaluate the model
y_pred = knn.predict(X_test)
acc2 = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc2}')

# Step 3.3.1: Train a Decision Tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Step 3.3.2: Evaluate the model
y_pred = dtc.predict(X_test)
acc3 = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc3}')

# Step 5: Save the trained model
if acc1 > acc2 and acc1 > acc3:
    print('Random Forest Classifier model loaded successfully')
    joblib.dump(clf, 'modelok.pkl')
elif acc2 > acc1 and acc2 > acc3:
    print('KNN model loaded successfully')
    joblib.dump(knn, 'modelok.pkl')
elif acc3 > acc1 and acc3 > acc2:
    print('Decision Tree Classifier model loaded successfully')
    joblib.dump(dtc, 'modelok.pkl')
else:
    print('No model suits the data')
"""
# Step 5: Real-time Detection with Probability and Landmark Visualization
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract the same points used during training
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
        
        # Flatten the points and make predictions
        feature_vector = np.array(points).flatten().reshape(1, -1)
        prediction = clf.predict(feature_vector)
        prediction_prob = clf.predict_proba(feature_vector)

        # Show the prediction probabilities for faint, sitting, and standing
        prob_faint = prediction_prob[0][1] * 100
        prob_sitting = prediction_prob[0][2] * 100
        prob_standing = prediction_prob[0][3] * 100
        
        # Display probabilities on the frame
        text_faint = f'Faint Probability: {prob_faint:.2f}%'
        text_sitting = f'Sitting Probability: {prob_sitting:.2f}%'
        text_standing = f'Standing Probability: {prob_standing:.2f}%'

        cv2.putText(frame, text_faint, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text_sitting, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text_standing, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Display the detected class with highest probability
        if prob_faint > 90:
            cv2.putText(frame, 'Faint Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            whatsapp_number = "+917204677562"
            alert_message = "Fainting detected! Please check immediately."
            send_whatsapp_alert(whatsapp_number, alert_message)
            consecutive_frames = 0  # Reset counter after sending alert

        elif prediction == 2:
            cv2.putText(frame, 'Sitting Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        elif prediction == 3:
            cv2.putText(frame, 'Standing Detected!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw landmarks on the frame for better understanding
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the output frame
    cv2.imshow('Faint, Sitting, and Standing Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""