# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:53:25 2023

@author: 001ku
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import timeit
import os
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 720)
folderPath = "Presentation"
##start time ######################################

startTime = timeit.default_timer()
sTime=(startTime / 1000) % 60
print(sTime,'seconds')
#############################################################
detector = HandDetector(detectionCon=0.9)

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
# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)


startDist = None
scale = 0
cx, cy = 500,500
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    pathImages= cv2.imread(folderPath)
    if len(hands) == 2:
        print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1]))
        
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            print("Zoom Gesture")
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            # point 8 is the tip of the index finger
            if startDist is None:
                #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

                startDist = length

            #length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print(scale)
    else:
        if len(hands) == 1 and buttonPressed is False:
            print(detector.fingersUp(hands[0]))
            fingers = detector.fingersUp(hands[0])
            if fingers == [1, 0, 0, 0, 0]:
                print("Zoom Gesture")
                lmList1 = hands[0]["lmList"]
                fingers = detector.fingersUp(hands[0]) 
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
     
    try:
        h1, w1, _= pathImages.shape
        newH, newW = ((h1+scale)//2)*2, ((w1+scale)//2)*2
        pathImages = cv2.resize(pathImages, (newW,newH))

        img[cy-newH//2:cy+ newH//2, cx-newW//2:cx+ newW//2] = pathImages
    except:
        pass
    
    
    #return

    
                
                
                
                
                
    cv2.imshow("Image", img)
    if (cv2.waitKey(30)== 27):
        break
    
    #end time ###
endTime = timeit.default_timer()

print(endTime - startTime,'seconds')