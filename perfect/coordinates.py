# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:11:30 2023

@author: 001ku
"""

import cv2

# Load the hand detection model
hand_cascade = cv2.CascadeClassifier('C:/Users/001ku/OneDrive/Desktop/perfect/hand.xml/hand.xml')

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the hand in the frame
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around the detected hand(s)
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(hands)
    # Display the frame with the detected hand(s)
    cv2.imshow('Hand detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
