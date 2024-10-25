import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import warnings

# Suppress warnings from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load the dataset
dataset_path = 'shoulder_exercise_dataset.csv'
data = pd.read_csv(dataset_path)

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

# Function to check if the angle is within the dataset's range
def is_angle_correct(angle, angle_type):
    angle_data = data[f'{angle_type}_Angle'].values
    differences = np.abs(angle_data - angle)
    min_difference = np.min(differences)
    return min_difference < 15  # You can adjust this threshold as needed

def getside(angle):
    if(angle>=35 and angle <=40):
        return "up"
    elif(angle>=75 and angle <=80):
        return "down"

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open video capture
cap = cv2.VideoCapture(0)

# Set up the Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    workout_count = 0
    rep_direction = None
    rep_in_progress = False  # Flag to track if a repetition is in progress
    correct_up_position = False
    correct_down_position = False
    instruction = "" 
    counter = 0
    stage = "up"
    f = 1 # Instruction for the user (either "Move Up" or "Move Down")

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
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Check if angles are correct
                right_correct = is_angle_correct(right_elbow_angle, 'Right')
                left_correct = is_angle_correct(left_elbow_angle, 'Left')
                

                if right_correct and left_correct:
                    feedback = "Correct Form"
                    color = (0, 255, 0)  # Green for correct form
                else:
                    f = 0
                    feedback = "Incorrect Form"
                    color = (0, 0, 255)
                
                mov=getside(right_elbow_angle)


                if mov=='down' and stage == "up":
                    stage = "down"
                
                if mov=='up' and stage == "down":
                    stage = "up"
                    if f == 1:
                        counter += 1
                    else:
                        f = 1
                if counter == 5:
                    feedback = "You successfully completed your target"
                
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
        cv2.imshow('Shoulder Exercise Form Correction', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()
