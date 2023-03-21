# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:26:35 2023

@author: 001ku
"""

from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import time



# Parameters
width, height = 420, 420
center = 200

folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, width)
cap.set(4, height)

############################start time#########################################

start_time = time.time()
sTime=((start_time / 1000) % 60)
print(sTime,'seconds')
###############################################################################
# Hand Detector
detectorHand = HandDetector(detectionCon=0.7)

# Variables
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image
startDist = None
scale = 0


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
# Initialize previous centroid position
prev_centroid = None
alpha = 0.5
kernel_size = 15

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    hands, img = detectorHand.findHands(img)  # with draw
    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Threshold the image to get the hand region
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw Gesture Threshold line
    #cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)
    # Find the hand and its landmarks
    
    
    
    if len(contours) > 0:
            hand_contour = max(contours, key=cv2.contourArea)
            # Calculate centroid of hand contour
            M = cv2.moments(hand_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
# Draw center point on frame
    #cv2.circle(img, centroid, 5, (0, 255, 0), -1)
    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        cv2.circle(img, (int(cx), int(cy)), 10, (0, 255, 0), cv2.FILLED)
        fingers = detectorHand.fingersUp(hand) # If hand is detected
        if prev_centroid is not None:   
        # Calculate the displacement from the previous position
            dx = cx - prev_centroid[0]
            dy = cy - prev_centroid[1]
            if abs(dx) > abs(dy):
                if hands and buttonPressed is False:
                
                    if fingers == [1, 1, 1, 1, 1] and prev_centroid[0]< centroid[0]:
                        cv2.putText(img, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                        print("Left")
                
                        buttonPressed = True
                        if imgNumber > 0:
                            imgNumber -= 1
                            annotations = [[]]
                            annotationNumber = -1
                            annotationStart = False
                    if fingers == [1, 1, 1, 1, 1] and prev_centroid[0]>centroid[0]:
                        cv2.putText(img, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                        print("Right")
                        buttonPressed = True
                        if imgNumber < len(pathImages) - 1:
                            imgNumber += 1
                            annotations = [[]]
                            annotationNumber = -1
                            annotationStart = False         
                else:
                    annotationStart = False
        

                if buttonPressed:
                    counter += 1
                    if counter > delay:
                        counter = 0
                        buttonPressed = False

                for i, annotation in enumerate(annotations):
                    for j in range(len(annotation)):
                        if j != 0:
                            cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)
    
    #cv2.putText(img, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Predict the next state and update the filter
    prediction = kalman.predict()
    # Update previous centroid position
    prev_centroid = centroid
    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image",cv2.flip( img,1))
    #cv2.imshow("Slides1",gray )
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Get the ending time
end_time = time.time()
total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60

print("Total time taken: {:.2f} seconds or {:.2f} minutes".format(total_time_seconds, total_time_minutes))