import cv2
import mediapipe as mp
import numpy as np

# Define the color of the cube
CUBE_COLOR = (255, 0, 0)  # Blue

# Create a Hands object
hands = mp.solutions.hands.Hands()

# Start capturing the video
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the hands
    results = hands.process(frame)

    # Draw the cube at the centroid of the hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Compute the size of the cube based on the distance between the wrist and the middle finger base
            wrist_landmark = hand_landmarks.landmark[0]
            mf_base_landmark = hand_landmarks.landmark[9]
            dist = np.sqrt((wrist_landmark.x - mf_base_landmark.x)**2 + (wrist_landmark.y - mf_base_landmark.y)**2 + (wrist_landmark.z - mf_base_landmark.z)**2)
            cube_size = int(dist * 200)

            # Compute the position of the cube at the centroid of the hand
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            hand_landmarks_list = [0, 5, 9, 13, 17]  # Landmark indices for the wrist and finger bases
            centroid = np.mean(landmarks[hand_landmarks_list], axis=0)
            cx, cy, cz = centroid
            x1, y1 = int(cx * frame.shape[1] - cube_size/2), int(cy * frame.shape[0] - cube_size/2)
            x2, y2 = x1 + cube_size, y1 + cube_size
            z = cz * 100  # Scale the z-coordinate to match the size of the cube
    
            # Draw the cube
            cv2.rectangle(frame, (x1, y1), (x2, y2), CUBE_COLOR, -1)
            cv2.putText(frame, f'z={z:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CUBE_COLOR, 2)

    # Show the frame
    cv2.imshow('Hand Detection', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
