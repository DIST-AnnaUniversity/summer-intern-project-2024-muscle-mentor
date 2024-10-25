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

# Load the dataset with correct angles
df = pd.read_csv('squat_exercise_angles.csv')

# Define ranges for angles
min_hip_knee_angle = df['Hip_Knee_Angle'].min()  # Minimum angle for correct form
max_hip_knee_angle = df['Hip_Knee_Angle'].max()  # Maximum angle for correct form
min_knee_ankle_angle = df['Knee_Ankle_Angle'].min()
max_knee_ankle_angle = df['Knee_Ankle_Angle'].max()

# Define thresholds for detecting small angles
threshold_small_angle = 10  # Degrees near the min and max angles to detect a transition

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize variables for counting
counter = 0
stage = "down"  # Start in the down position
f = 1
correct_form_counter = 0  # Counter for correct forms

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                
                # Determine feedback and form validity
                if (min_hip_knee_angle <= hip_knee_angle <= max_hip_knee_angle) and \
                   (min_knee_ankle_angle <= knee_ankle_angle <= max_knee_ankle_angle):
                    feedback = "Correct Form"
                    color = (0, 255, 0)  # Green for correct
                    correct_form = True
                else:
                    feedback = "Incorrect Form"
                    color = (0, 0, 255)  # Red for incorrect
                    correct_form = False
                
                # Stage transitions and counting
                if hip_knee_angle <= min_hip_knee_angle + threshold_small_angle and stage == "down":
                    stage = "up"
                
                if hip_knee_angle >= max_hip_knee_angle - threshold_small_angle and stage == "up":
                    stage = "down"
                    if correct_form:
                        if f == 1:
                            counter += 1
                            correct_form_counter += 1
                            f = 0
                        else:
                            f = 1
                else:
                    correct_form_counter = 0  # Reset the correct form counter if form is incorrect
                    
                if counter == 10:
                    feedback = "You successfully completed your target"
                
                # Display feedback, counter, and stage
                cv2.putText(image, stage, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(image, f'Reps: {counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
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

# Release the capture
cap.release()
cv2.destroyAllWindows()
