import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings

# Suppress warnings from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point
    
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

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_count = 0  # Frame counter
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
                
                # Get coordinates for squat exercise
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate angles
                hip_knee_angle = calculate_angle(hip, knee, ankle)
                knee_ankle_angle = calculate_angle(knee, ankle, [ankle[0], ankle[1] + 0.1])  # Slightly adjusted point for ankle
                
                # Visualize angles
                cv2.putText(image, f'Hip-Knee: {round(hip_knee_angle, 2)}', 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, f'Knee-Ankle: {round(knee_ankle_angle, 2)}', 
                            tuple(np.multiply(ankle, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Store data in list
                data_list.append({
                    'Frame': frame_count,
                    'Hip_X': hip[0],
                    'Hip_Y': hip[1],
                    'Knee_X': knee[0],
                    'Knee_Y': knee[1],
                    'Ankle_X': ankle[0],
                    'Ankle_Y': ankle[1],
                    'Hip_Knee_Angle': hip_knee_angle,
                    'Knee_Ankle_Angle': knee_ankle_angle
                })

                frame_count += 1

            except Exception as e:
                print(f"An error occurred: {e}")
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Display the resulting frame
        cv2.imshow('Squat Exercise Form Correction', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Save the DataFrame to a CSV file
data = pd.DataFrame(data_list)
data.to_csv('squat_exercise_angles.csv', index=False)

# Release the capture
cap.release()
cv2.destroyAllWindows()
