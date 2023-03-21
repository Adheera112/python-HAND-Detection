# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:46:01 2023

@author: 001ku
"""

from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import time


# Parameters
width, height = 280, 420
gestureThreshold = 300

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
startDist = None
scale = 0
# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    #cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                cv2.putText(img, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                print("Left")
                
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 1, 0, 0, 0]:
                cv2.putText(img, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            cv2.putText(img, 'Click', org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            print("click")
            
        if fingers == [0, 0, 0, 0, 1]:
            cv2.putText(img, 'Write', org, font, fontScale, color, thickness, cv2.LINE_AA)
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [1, 1, 1, 1, 1]:
            cv2.putText(img, 'Erase', org, font, fontScale, color, thickness, cv2.LINE_AA)
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

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
    
    
    width, height = 420, 720
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall
    
    cx, cy = 500,500
    if len(hands) == 2:
        cv2.putText(img, 'Zoom', org, font, fontScale, color, thickness, cv2.LINE_AA)
        print(detectorHand.fingersUp(hands[0]), detectorHand.fingersUp(hands[1]))
        
        if detectorHand.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
               detectorHand.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            print("Zoom Gesture")
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            # point 8 is the tip of the index finger
            if startDist is None:
                #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                length, info, img = detectorHand.findDistance(hands[0]["center"], hands[1]["center"], img)

                startDist = length

            #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
            length, info, img = detectorHand.findDistance(hands[0]["center"], hands[1]["center"], img)

            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print(scale)
    else:
        startDist = None

    try:
        imgSmall = cv2.resize(img, (ws, hs))
        h, w, _= imgCurrent.shape
        newH, newW = ((hs+scale)//2)*2, ((ws+scale)//2)*2
        imgCurrent= cv2.resize(imgCurrent, (newW,newH))
        imgCurrent[cy-newH//2:cy+ newH//2, cx-newW//2:cx+ newW//2] = imgSmall
        
    except:
        pass
    
    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Get the ending time
end_time = time.time()
total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60

print("Total time taken: {:.2f} seconds or {:.2f} minutes".format(total_time_seconds, total_time_minutes))