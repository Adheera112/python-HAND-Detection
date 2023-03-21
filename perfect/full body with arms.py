# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:09:55 2023

@author: HP
"""

import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()
cap = cv2.VideoCapture(0)

while True:
    rat , img = cap.read()
    img = detector.findPose(img)
    lmlist , bboxInfo = detector.findPosition(img)
    cv2.imshow("result",cv2.flip(img,1))
    if(cv2.waitKey(30)==27):
        break
    
cap.release()
cv2.destroyAllWindows() 