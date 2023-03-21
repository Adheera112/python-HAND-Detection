# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:10:28 2023

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
index= (120,0)
# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    hands, img = detectorHand.findHands(img)  # with draw
    
                
    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"] 
        cv2.circle(img, (int(cx), int(cy)), 10, (0, 255, 0), cv2.FILLED)
        fingers = detectorHand.fingersUp(hand) # If hand is detected
        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal  ,yVal
        print(indexFinger)
        
        if hands and buttonPressed is False:
            if fingers == [1, 1, 1, 1, 1] and index > indexFinger :
                cv2.putText(img, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                     imgNumber -= 1
                     annotations = [[]]
                     annotationNumber = -1
                     annotationStart = False
            if fingers == [1, 1, 1, 1, 1] and index < indexFinger :
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
            if counter <= delay:
                counter = 0
                buttonPressed = False

        for i, annotation in enumerate(annotations):
            for j in range(len(annotation)):
                if j != 0:
                    cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)
                    
                    
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
        