# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:05:39 2023

@author: 001ku
"""

import cv2

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Define background subtractor object
bgsub = cv2.createBackgroundSubtractorMOG2()

# Initialize previous centroid position
prev_centroid = None

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    
    # Apply background subtraction to get hand region
    hand_mask = bgsub.apply(frame)
    hand_mask = bgsub.apply(frame)
    # Apply contour detection to extract hand contour
    contours, hierarchy = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)
        # Calculate centroid of hand contour
        M = cv2.moments(hand_contour)
        if M['m00'] > 0:
            centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            # Draw center point on frame
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            # Calculate movement direction based on centroid position
            if prev_centroid is not None:
                if centroid[0] > prev_centroid[0]:
                    direction = 'Right'
                elif centroid[0] < prev_centroid[0]:
                    direction = 'Left'
                else:
                    direction = 'No movement'
                # Display movement direction on frame
                cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Update previous centroid position
            prev_centroid = centroid
    
    # Display the resulting frame
    cv2.imshow('Hand tracking', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
