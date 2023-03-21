# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:14:01 2023

@author: 001ku
"""

from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np



# Initialize hand detector with detection confidence threshold of 0.7
detector = HandDetector(detectionCon=0.7)

# Open the default camera
cap = cv2.VideoCapture(0)

prev_pos = None
alpha = 0.5
kernel_size = 15
# Loop through frames from the camera
while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break
    

   # Find the hand in the frame
    hands, _ = detector.findHands(img)

# If a hand is detected, get its centroid
    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        if prev_pos is not None:
            # Calculate the displacement from the previous position
            dx = cx - prev_pos[0]
            dy = cy - prev_pos[1]
            # Check if the hand moves left or right
            if abs(dx) > abs(dy):
                if dx > 0:
                    direction = "right"
                else:
                    direction = "left"
            else:
                direction = None
        # Apply a filter to smooth out the movement
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
            cx = cv2.filter2D(np.float32([cx]), -1, kernel)[0]
            cy = cv2.filter2D(np.float32([cy]), -1, kernel)[0]
    # Draw a circle at the centroid position
        cv2.circle(img, (int(cx), int(cy)), 10, (0, 255, 0), cv2.FILLED)
    # Store the current centroid position for the next frame
        prev_pos = (cx, cy)


    # Show the frame
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) == 27:  # Esc key to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
