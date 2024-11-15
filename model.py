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
not_fainting = extract_dataset_features('not_fainting', 0)
sitting_videos = extract_dataset_features('sitting', 2)
standing_videos = extract_dataset_features('standing', 3)
walking_videos = extract_dataset_features('walking', 4)

# Combine all features and labels
dataset = faint_videos + not_fainting + sitting_videos + standing_videos

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
print(f'Random Forest Classifier Accuracy: {acc1}')

# Step 3.2.1: Train a KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Step 3.2.2: Evaluate the model
y_pred = knn.predict(X_test)
acc2 = accuracy_score(y_test, y_pred)
print(f'KNN Classifier Accuracy: {acc2}')

# Step 3.3.1: Train a Decision Tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Step 3.3.2: Evaluate the model
y_pred = dtc.predict(X_test)
acc3 = accuracy_score(y_test, y_pred)
print(f'Decision Tree Classifier Accuracy: {acc3}')

# Step 5: Save the trained model
if acc1 >= acc2 and acc1 >= acc3:
    print('Random Forest Classifier model loaded successfully')
    joblib.dump(clf, 'modelok.pkl')
elif acc2 >= acc1 and acc2 >= acc3:
    print('KNN model loaded successfully')
    joblib.dump(knn, 'modelok.pkl')
elif acc3 >= acc1 and acc3 >= acc2:
    print('Decision Tree Classifier model loaded successfully')
    joblib.dump(dtc, 'modelok.pkl')
else:
    print('No model suits the data')
