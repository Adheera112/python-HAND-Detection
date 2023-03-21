# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:01:55 2023

@author: 001ku
"""
from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import time

# Parameters
width, height = 720, 560
folderPath = "Presentation"

# Camera Setup
with cv2.VideoCapture(0) as cap:
    cap.set(3, width)
    cap.set(4, height)

    # Hand Detector
    detectorHand = HandDetector(detectionCon=0.8,minTrackCon=0.7,maxHands=1)

    # Variables b 
    org = (40, 40)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 100
    prev_position = '' # initialize hand position flag

    # Get list of presentation images
    resizedImages = [cv2.resize(cv2.imread(os.path.join(folderPath, imagePath)), (720, 420)) for imagePath in sorted(os.listdir(folderPath), key=len)]

    # Start time
    start_time = time.time()
    sTime = ((start_time / 1000) % 60)
    print(sTime, 'seconds')

    with open("C:/Users/001ku/OneDrive/Desktop/perfect/abc.txt", 'a') as file:
        for i, _ in enumerate(resizedImages):
            # Get image frame
            success, img = cap.read()
            flipped_img = cv2.flip(img, 1)
            flipped_imgCurrent = resizedImages[i]

            # Get the center of the screen
            center_x, center_y = int(flipped_img.shape[1]/2), int(flipped_img.shape[0]/2)

            # Draw a white circle with radius 50 and thickness 2
            circle = cv2.circle(flipped_img, (center_x, center_y), 20, (255, 255, 255), 2)

            # Convert the frame to grayscale and blur it
            gray = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # Threshold the image to get the hand region
            thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hands, flipped_img = detectorHand.findHands(flipped_img)  # with draw
            if hands:
                hand = hands[0]
                cx, cy = hand["center"]
                cx = int(cx)
                cy = int(cy
                #cv2.circle(flipped_img, (int(cx), int(cy)), 10, (0, 255, 0), 15)
                # Calculate distance between hand center and screen center
                dist_x = cx - flipped_img.shape[1] // 2
                dist_y = cy - flipped_img.shape[0] // 3
                displacement = ((dist_x)**2 + (dist_y)**2)**0.5
        
                if len(contours) > 0:
                    cnt = max(contours, key=cv2.contourArea)
                    M = cv2.moments(cnt)
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
        
                 # Draw a vertical line in the middle of the frame
                    cv2.line(flipped_img, (flipped_img.shape[1] // 2, 0), (flipped_img.shape[1] // 2, flipped_img.shape[0]), (0, 255, 0), 2)
                # Draw two horizontal lines dividing the frame into three parts
                     #cv2.line(flipped_img, (0, flipped_img.shape[0] // 3), (flipped_img.shape[1], flipped_img.shape[0] // 3), (0, 255, 0), 2)
                     #cv2.line(flipped_img, (0, 2 * flipped_img.shape[0] // 3), (flipped_img.shape[1], 2 * flipped_img.shape[0] // 3), (0, 255, 0), 2)  
                    if displacement < threshold:
                        fingers = detectorHand.fingersUp(hand)
                    else:
            # Hand is moving
                        fingers = detectorHand.fingersUp(hand)
                        if fingers==[1,1,1,1,1]:
                            if cx < flipped_img.shape[1] // 2:
                                # Hand is on the left side of the screen
                                    if prev_position != 'left':
                                        cv2.putText(flipped_img, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                        print("Left")
                                        image_index = max(image_index-2, 0)
                                        prev_position = 'left'
                            elif cx > flipped_img.shape[1] // 2:
                                
                                    # Hand is on the right side of the screen
                                    if prev_position != 'right':
                                        cv2.putText(flipped_img, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                        print("Right")
                                        image_index = min(image_index+2, len(pathImages)-1)
                                        prev_position = 'right'
                            prev_cx, prev_cy = cx, cy
                            
                    
                    if displacement > threshold1:
                        fingers = detectorHand.fingersUp(hand)
                    else:
                        # Hand is moving
                        fingers = detectorHand.fingersUp(hand)
                        if fingers!=[0,0,0,0,0]:
                            
                            if cy > 2 * flipped_img.shape[0] // 3:
                                # Hand is below the lower line
                                if prev_position != 'down':
                                    cv2.putText(flipped_img, 'Down', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                    print("Down")
                                    #Move image down by 10 pixels
                                    #y_offset += 10
                                    prev_position = 'down'
                            elif cy < flipped_img.shape[0] // 3:
                                # Hand is above the upper line
                                if prev_position != 'up':
                                    cv2.putText(flipped_img, 'Up', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                    print("Up")
                                    # Move image up by 10 pixels
                                    #y_offset -= 10
                                    prev_position = 'up'
                        prev_cx, prev_cy = cx, cy
                    
            cv2.flip(flipped_img,1)
            cv2.imshow("Slides", flipped_imgCurrent)
            cv2.imshow("Image", flipped_img) 
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