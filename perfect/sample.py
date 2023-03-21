# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:53:25 2023

@author: 001ku
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import timeit
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
##start time ######################################

startTime = timeit.default_timer()
sTime=(startTime / 1000) % 60
STime=(startTime / 1000) / 60
print(sTime,'seconds')
#############################################################
detector = HandDetector(detectionCon=0.9)
startDist = None
scale = 0
cx, cy = 500,500
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread("C:/Users/001ku/OneDrive/Desktop/perfect/images.jpeg")
    img2 = cv2.imread("C:/Users/001ku/OneDrive/Desktop/perfect/download.jpeg")
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
        if len(hands) == 1:
            print(detector.fingersUp(hands[0]))
            
            if detector.fingersUp(hands[0]) == [0, 1, 1, 0, 0]:
                print("Zoom Gesture")
                #print("Image", img2)
                
    try:
        h1, w1, _= img1.shape
        newH, newW = ((h1+scale)//2)*2, ((w1+scale)//2)*2
        img1 = cv2.resize(img1, (newW,newH))

        img[cy-newH//2:cy+ newH//2, cx-newW//2:cx+ newW//2] = img1
    except:
       pass
       
    cv2.imshow("Image", img)
    if (cv2.waitKey(30)== 27):
        break
    
    #end time ###
endTime = timeit.default_timer()

print(endTime - startTime,'seconds')