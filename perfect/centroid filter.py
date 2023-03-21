# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:20:35 2023

@author: 001ku
"""

import numpy as np
import cv2

# Define the state vector and measurement vector
state = np.zeros((2, 1))
measurement = np.zeros((1, 1))

# Define the state transition matrix and measurement matrix
dt = 0.1
A = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])

# Define the process noise and measurement noise
Q = np.array([[0.01, 0], [0, 0.01]])
R = np.array([[0.1]])

# Initialize the filter
P = np.eye(2) * 100
x = np.zeros((2, 1))
v = np.zeros((2, 1))
kalman = cv2.KalmanFilter(2, 1)
kalman.transitionMatrix = A
kalman.measurementMatrix = H
kalman.processNoiseCov = Q
kalman.measurementNoiseCov = R
kalman.statePost = x
kalman.errorCovPost = P

# Capture the video
cap = cv2.VideoCapture(0)

# Loop over each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Threshold the image to get the hand region
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the centroid of the hand region
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            measurement[0, 0] = cx
        else:
            measurement[0, 0] = state[0, 0]
    else:
        measurement[0, 0] = state[0, 0]
    
    # Predict the next state and update the filter
    prediction = kalman.predict()
    cv2.imshow("Image",cv2.flip( frame,1))
#cv2.imshow("Slides1",gray )
    key = cv2.waitKey(1)
    if key == ord('q'):
        break