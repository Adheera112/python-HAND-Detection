# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:32:28 2023

@author: 001ku
"""

from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import time
import mediapipe as mp



# Parameters
width, height = 280, 420
gestureThreshold = 300
folderPath = "Presentation"


# Initialize the video stream
cap = cv2.VideoCapture(0)

# Initialize the mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

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
# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)


while True:
    # Read a frame from the video stream
    success, image = cap.read()

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(image)

    # Extract the x-coordinate of the tip of the index finger
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = index_finger_tip.x * image.shape[1]

            # Map the x-coordinate to movement commands
            if x < image.shape[1] / 3:
                print("Move left")
            elif x > 2 * image.shape[1] / 3:
                print("Move right")
            else:
                print("Don't move")

    # Display the image with hand landmarks
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hand tracking", image)

    # Exit the loop if the user presses the "q" key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()


