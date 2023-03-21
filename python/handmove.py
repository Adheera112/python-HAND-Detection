import cv2
import mediapipe as mp
import numpy as np

class Hand:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def detect(self, frame):
        # Convert the BGR frame to RGB for Mediapipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)

        # Detect hands in the RGB frame
        results = self.hands.process(frame)

        # Draw the hand landmarks on the RGB frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Get the x, y, and z coordinates of the middle finger tip landmark (Landmark 8)
                x = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1]
                y = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0]
                z = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                # Round the x, y, and z coordinates to two decimal places
                x = round(x/100,0)
                y = round(y/100,0)
                z = round(z/100,0)

                # Print the x, y, and z coordinates in pixel coordinates
                print(f"{x},{y},{z}")

                # Send hand position coordinates to other application using UDP communication

                return x

        # If hand is not detected, print x, y, and z as 0
        else:
            x, y, z = 0, 0, 0
            print(" 0,0,0")
            return x


class HandMovementDetector:
    def __init__(self, buffer_size=10, threshold=1):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.prev_x = None
        self.x_buffer = []

    def detect(self, x):
        # Add the x-coordinate to the buffer
        self.x_buffer.append(x)

        # If the buffer is full, remove the oldest value
        if len(self.x_buffer) > self.buffer_size:
            self.x_buffer.pop(0)

        # Compute the moving average of the x-coordinates in the buffer
        if len(self.x_buffer) == self.buffer_size:
            x_avg = np.mean(self.x_buffer)

            # Check for hand movement
            if self.prev_x is not None and abs(x_avg - self.prev_x) < self.threshold:
                if x_avg < self.prev_x:
                    direction = -1
                    print("left")
                elif x_avg > self.prev_x:
                    direction = 1
                    print("right")
                else:
                    direction = 0

                # Store the current moving average as the previous x-coordinate for the next frame
                self.prev_x = x_avg


             

