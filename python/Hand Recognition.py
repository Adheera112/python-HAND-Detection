import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.5, maxHands=2)

while True:
    ret, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        for i, hand in enumerate(hands):
            lmList = hand["lmList"]
            bbox = hand["bbox"]
            handType = hand["type"]
            fingers = detector.fingersUp(hand)
            height1, _, _ = detector.findDistance(lmList[5][0:2], lmList[6][0:2], img)
            height2, _, _ = detector.findDistance(lmList[6][0:2], lmList[7][0:2], img)
            height3, _, _ = detector.findDistance(lmList[7][0:2], lmList[8][0:2], img)
            height = height1 + height2 + height3
            width, _, _ = detector.findDistance(lmList[7][0:2], lmList[8][0:2], img)
            dist = tuple(round(num, 1) for num in (height, width))
            color = (0, 0, 0) if i == 0 else (0, 0, 0)
            cv2.putText(img, f'Index Finger{i+1}: {handType} - {dist}mm - {fingers}', (10, 30+i*30), cv2.FONT_HERSHEY_PLAIN, 1.12, color, 2)

    cv2.imshow("Final Output", img)
    if cv2.waitKey(1) == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()