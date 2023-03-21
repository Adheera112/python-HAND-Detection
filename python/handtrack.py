from HandTrackingModule import HandDetector
import cv2
import os
import time
import mediapipe

# Parameters
width, height = 300,300
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.7,minTrackCon=0.7, maxHands=1)

# Variables b 
org = (40, 40)
fontScale = 1
color = (255, 0, 0)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
image_index = 0
threshold = 50
threshold1=200# define threshold value
prev_position = '' # initialize hand position flag

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
resizedImages = [cv2.resize(cv2.imread(os.path.join(folderPath, imagePath)), (720, 420)) for imagePath in pathImages]

# Start time
start_time = time.time()
sTime=((start_time / 1000) % 60)
print(sTime,'seconds')

with open("C:/Users/IF 01/Desktop/kunj/abc.txt", 'a') as file:
    while True:
        # Get image frame
        success, img = cap.read()
        pathFullImage = os.path.join(folderPath, pathImages[image_index])
        imgCurrent = cv2.imread(pathFullImage)
        # Get the center of the screen
        #center_x, center_y = int(img.shape[1]/2), int(img.shape[0]/2)
        
        # Draw a white circle with radius 50 and thickness 2
        #circle=cv2.circle(img, (center_x, center_y), 20, (255, 255, 255), 2)
        #cv2.putText(circle, 'Place Your Hand At circle', org, font, fontScale, color, thickness, cv2.LINE_AA)
        # Convert the frame to grayscale and blur it
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
        # Threshold the image to get the hand region
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hands, img = detectorHand.findHands(img)  # with draw
        if hands:
            hand = hands[0]
            #lmList = hand["lmList"] 
            cx, cy = hand["center"]
            cx = int(cx)
            cy = int(cy)
            ce = (cx,cy)
            list= []
            for i in range(len(list)+1):
                list.append(ce)
            print(list)
            file.write(f"({cx},{cy})\n")
    
            #cv2.circle(img, (int(cx), int(cy)), 10, (0, 255, 0), 15)
            # Calculate distance between hand center and screen center
            dist_x = cx - img.shape[1] // 2
            dist_y = cy - img.shape[0] // 3
            displacement = ((dist_x)**2 + (dist_y)**2)**0.5
    
            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
    
                 # Draw a vertical line in the middle of the frame
                #cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (0, 255, 0), 2)
                # Draw two horizontal lines dividing the frame into three parts
                #cv2.line(img, (0, img.shape[0] // 3), (img.shape[1], img.shape[0] // 3), (0, 255, 0), 2)
                #cv2.line(img, (0, 2 * img.shape[0] // 3), (img.shape[1], 2 * img.shape[0] // 3), (0, 255, 0), 2)  
                
                
                if displacement < threshold:
                    fingers = detectorHand.fingersUp(hand)
                    if prev_position != 'center':
                        cv2.putText(img, 'Center', org, font, fontScale, color, thickness, cv2.LINE_AA)
                        #print("Center")
                        #image_index = len(pathImages) // 2
                        prev_position = 'center'
                else:
                    fingers = detectorHand.fingersUp(hand)
                    if fingers==[1,1,1,1,1]:
                        if cx < img.shape[1] // 2:
                            # Hand is on the left side of the screen
                             if prev_position != 'left':
                                 cv2.putText(img, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                 print("Left")
                                 image_index = max(image_index-1, 0)
                                 prev_position = 'left'
                        elif cx > img.shape[1] // 2: 
                            # Hand is on the right side of the screen
                            if prev_position != 'right':
                                cv2.putText(img, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                print("Right")
                                image_index = min(image_index+1, len(pathImages)-1)
                                prev_position = 'right'
                    prev_cx, prev_cy = cx, cy
    
                if displacement > threshold1:
                    fingers = detectorHand.fingersUp(hand)
    
                else:
                    fingers = detectorHand.fingersUp(hand)
                    if fingers!=[0,0,0,0,0]:
                        if cy > 2 * img.shape[0] // 3:
                            # Hand is below the lower line
                            if prev_position != 'down':
                                cv2.putText(img, 'Down', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                print("Down")
                                #Move image down by 10 pixels
                                #y_offset += 10
                                prev_position = 'down'
                        elif cy < img.shape[0] // 3:
                            # Hand is above the upper line
                            if prev_position != 'up':
                               cv2.putText(img, 'Up', org, font, fontScale, color, thickness, cv2.LINE_AA)
                               print("Up")
                               # Move image up by 10 pixels
                               #y_offset -= 10
                               prev_position = 'up'
                    prev_cx, prev_cy = cx, cy
                
        #cv2.flip(img,1)
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






