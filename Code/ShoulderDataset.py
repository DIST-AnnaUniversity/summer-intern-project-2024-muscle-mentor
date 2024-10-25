import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings

# Suppress warnings from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open video capture
cap = cv2.VideoCapture(0)

# Create an empty list to store data
data_list = []

# Set up the Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_count = 0  # Frame counter
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            try:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Right arm coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Left arm coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angles
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Store data in list
                data_list.append({
                    'Frame': frame_count,
                    'Right_Shoulder_X': right_shoulder[0],
                    'Right_Shoulder_Y': right_shoulder[1],
                    'Right_Elbow_X': right_elbow[0],
                    'Right_Elbow_Y': right_elbow[1],
                    'Right_Wrist_X': right_wrist[0],
                    'Right_Wrist_Y': right_wrist[1],
                    'Right_Angle': right_angle,
                    'Left_Shoulder_X': left_shoulder[0],
                    'Left_Shoulder_Y': left_shoulder[1],
                    'Left_Elbow_X': left_elbow[0],
                    'Left_Elbow_Y': left_elbow[1],
                    'Left_Wrist_X': left_wrist[0],
                    'Left_Wrist_Y': left_wrist[1],
                    'Left_Angle': left_angle
                })

                frame_count += 1

            except Exception as e:
                print(f"An error occurred: {e}")
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Display the resulting frame
        cv2.imshow('Shoulder Exercise Form Data Collection', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Save the DataFrame to a CSV file
if data_list:
    data = pd.DataFrame(data_list)
    data.to_csv('shoulder_exercise_dataset.csv', index=False)
else:
    print("No data to save.")

# Release the capture
cap.release()
cv2.destroyAllWindows()
