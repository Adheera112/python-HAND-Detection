# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:51:05 2023

@author: 001ku
"""

import cv2
import math
import mediapipe as mp

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Initialize the mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    # Read a frame from the video stream
    success, image = cap.read()

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(image)

    # Extract the landmarks of interest
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the angle between the wrist, the base of the thumb, and the tip of the index finger
            v1 = [thumb.x - wrist.x, thumb.y - wrist.y]
            v2 = [index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y]
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = dot_product / (math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2))
            angle = math.degrees(math.acos(cos_angle))

            # Display the angle and the coordinates of the hand
            cv2.putText(image, f"Angle: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"X: {int(index_finger_tip.x * image.shape[1])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Y: {int(index_finger_tip.y * image.shape[0])}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image with hand landmarks
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hand tracking", image)

    # Exit the loop if the user presses the "q" key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2
