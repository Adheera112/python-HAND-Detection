# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:16:20 2023

@author: 001ku
"""

import cv2
import mediapipe as mp

# Initialize the hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Parameters
width, height = 640, 480

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    # Get image frame
    success, img = cap.read()
    
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find the hand landmarks
    results = hands.process(img)

    # Draw the landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(hand_landmarks)

    # Convert the image back to BGR for display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Hand Landmarks", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
