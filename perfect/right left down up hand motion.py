from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import time

# Parameters
width, height = 240, 240
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
image_index = 0

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
resizedImages = [cv2.resize(cv2.imread(os.path.join(folderPath, imagePath)), (720, 420)) for imagePath in pathImages]

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
    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Threshold the image to get the hand region
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hands:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"] 
        #cv2.circle(img, (int(cx), int(cy)), 10, (0, 255, 0), 20)
    # If there is a contour, calculate its centroid
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        
        # Draw a vertical line in the middle of the frame
            #cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (0, 255, 0), 2)
        # Draw a horizontal line in the middle of the frame
            #cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (0, 255, 0), 2)
            
            fingers = detectorHand.fingersUp(hand)
            if fingers == [1, 1, 1, 1, 1]:
                if cx > img.shape[1] // 2 :
                    cv2.putText(img, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    print("Left")
                    image_index = max(image_index-1, 0)
                elif cx < img.shape[1] // 2:
                    cv2.putText(img, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    print("Right")
                    image_index = min(image_index+1, len(pathImages )-1)
                    
            # Draw a horizontal line in the middle of the frame
            #cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (0, 255, 0), 2)  
            if cy < img.shape[0] // 2:
                    cv2.putText(img, 'UP', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    print("up")
                    image_index = max(image_index-1, 0)
                    
            elif cy > img.shape[0] // 2:
                    cv2.putText(img, 'DOWN', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    print("down")
                    image_index = min(image_index+1, len(pathImages )-1)

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






