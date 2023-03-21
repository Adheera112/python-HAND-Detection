# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:37:38 2023

@author: 001ku
"""

from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import time

# Parameters
width, height = 420, 420
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8)

# Variables
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
index = 150
image_index = 0

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
resizedImages = [cv2.resize(cv2.imread(os.path.join(folderPath, imagePath)), (420, 420)) for imagePath in pathImages]

# Start time
start_time = time.time()
sTime=((start_time / 1000) % 60)
print(sTime,'seconds')

while True:
    # Get image frame
    success, img = cap.read()
    pathFullImage = os.path.join(folderPath, pathImages[image_index])
    imgCurrent = cv2.imread(pathFullImage)
    hands, img = detectorHand.findHands(img)  # with draw
    
    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]
        cv2.circle(img, (int(cx), int(cy)), 10, (0, 255, 0), cv2.FILLED)
        fingers = detectorHand.fingersUp(hand)
        xVal = int(np.interp(lmList[8][0], [0, width], [-width // 2, width // 2]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [-height // 2, height // 2]))
        indexFinger = xVal
        if fingers == [1, 1, 1, 1, 1]:
            if index < indexFinger:
                cv2.putText(img, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                print("Left")
                image_index = max(image_index-1, 0)
            elif index > indexFinger:
                cv2.putText(img, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                print("Right")
                image_index = min(image_index+1, len(pathImages )-1)
            
            index=indexFinger

    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)
    #cv2.imshow("Slides1",gray )
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
# Release the resources
cap.release()
cv2.destroyAllWindows()

# Get the ending time
end_time = time.time()
total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60

print("Total time taken: {:.2f} seconds or {:.2f} minutes".format(total_time_seconds, total_time_minutes))






