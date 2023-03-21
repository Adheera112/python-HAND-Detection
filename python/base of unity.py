import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Initialize VideoCapture object
cap = cv2.VideoCapture(0)  # 0 for default camera, or pass the path to a video file

# Create a hands object
with mp_hands.Hands(
        max_num_hands=1,  # maximum number of hands to detect
        min_detection_confidence=0.5,  # minimum detection confidence
        min_tracking_confidence=0.5  # minimum tracking confidence
) as hands:
    prev_x = None
    x_buffer = []
    buffer_size = 10
    while cap.isOpened():
        # Read the frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB for Mediapipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the RGB frame
        results = hands.process(frame)

        # Draw the hand landmarks on the RGB frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the x-coordinate of the middle finger tip landmark (Landmark 8)
                x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1]

                # Add the x-coordinate to the buffer
                x_buffer.append(x)

                # If the buffer is full, remove the oldest value
                if len(x_buffer) > buffer_size:
                    x_buffer.pop(0)

                # Compute the moving average of the x-coordinates in the buffer
                if len(x_buffer) == buffer_size:
                    x_avg = np.mean(x_buffer)

                    # Compare the x-coordinate with the moving average to determine the direction
                    if prev_x is not None:
                        if x_avg > prev_x:
                            print("Hand is moving right")
                        elif x_avg < prev_x:
                            print("Hand is moving left")

                    # Store the current moving average as the previous x-coordinate for the next frame
                    prev_x = x_avg

        # Convert the RGB frame back to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
