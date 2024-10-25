import cv2
import mediapipe as mp
import numpy as np

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

# Set the range for the tricep exercise
min_allowed_angle = 80  # Adjust this angle based on the desired range for "down"
max_allowed_angle = 150  # Adjust this angle based on the desired range for "up"

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize variables for counting
counter = 0
stage = "up"
f = 1

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
        
        # Check if landmarks are detected
        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for tricep exercise
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                current_angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, str(round(current_angle, 2)), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Determine feedback
                if current_angle >= min_allowed_angle and current_angle <= max_allowed_angle:
                    feedback = "Correct Form"
                    color = (0, 255, 0)
                else:
                    f = 0
                    feedback = "Incorrect Form"
                    color = (0, 0, 255)
                
                # Stage transitions
                if current_angle < min_allowed_angle + 10 and stage == "up":
                    stage = "down"
                
                if current_angle > max_allowed_angle - 10 and stage == "down":
                    stage = "up"
                    if f == 1:
                        counter += 1
                    else:
                        f = 1
                        
                # Display feedback and counter
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
        cv2.imshow('Tricep Exercise Form Correction', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
# Release the capture
cap.release()
cv2.destroyAllWindows()
