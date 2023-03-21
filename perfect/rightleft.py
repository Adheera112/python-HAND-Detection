import cv2
from cvzone.HandTrackingModule import HandDetector
import time

wCam, hCam = 1280, 720


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = HandDetector(detectionCon=0.8, maxHands=2)

pTime =0
cTime = 0


while True:
    Success, img = cap.read()
    hands, img = detector.findHands(img)                    #With Draw
    
    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime = cTime
    
    
    if hands:
        
        # Hand 1
        
        hand1 = hands[0]
        lmList1 = hand1["lmList"]                          # List of 21 LandMarks points
        handType1 = hand1["type"]                          # Hand Type either Right or Left
        fingers1 = detector.fingersUp(hand1)
        
        
        
      
    cv2.putText(img,f'FPS: {int(fps)}',(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
    
    
    if cv2.waitKey(1) == ord("k"):
        break