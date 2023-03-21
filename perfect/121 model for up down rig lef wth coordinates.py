from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import time


# Parameters
width, height = 720, 560
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
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
image_index = 0
prev_cx, prev_cy = 0, 0
threshold = 100
threshold1=200# define threshold value
prev_position = '' # initialize hand position flag
last_direction = None
threshold_y=50




# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
resizedImages = [cv2.resize(cv2.imread(os.path.join(folderPath, imagePath)), (720, 420)) for imagePath in pathImages]

# Start time
start_time = time.time()
sTime=((start_time / 1000) % 60)
print(sTime,'seconds')

while True:
    # Get image frame
    success, image = cap.read()
    image = cv2.flip(image, 1)
    pathFullImage = os.path.join(folderPath, pathImages[image_index])
    imageCurrent = cv2.imread(pathFullImage)
    # Get the center of the screen
    center_x, center_y = int(image.shape[1]/2), int(image.shape[0]/2)
    
    # Draw a white circle with radius 50 and thickness 2
    circle=cv2.circle(image, (center_x, center_y), 20, (255, 255, 255), 2)
    #cv2.putText(circle, 'Place Your Hand At circle', org, font, fontScale, color, thickness, cv2.LINE_AA)
    # Convert the frame to grayscale and blur it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold the image to get the hand region
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hands, image = detectorHand.findHands(image)  # with draw
    if hands:
        hand = hands[0]
        #lmList = hand["lmList"] 
        cx, cy = hand["center"]
        cx = int(cx)
        cy = int(cy)
        
        
        
        with open("C:/Users/001ku/OneDrive/Desktop/perfect/abc.txt", 'a') as file:
            file.write(f"{cx},{cy}\n")

        #cv2.circle(image, (int(cx), int(cy)), 10, (0, 255, 0), 15)
        # Calculate distance between hand center and screen center
        dist_x = cx - image.shape[1] // 2
        dist_y = cy - image.shape[0] // 3
        displacement = ((dist_x)**2 + (dist_y)**2)**0.5

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

         # Draw a vertical line in the middle of the frame
            cv2.line(image, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), (0, 255, 0), 2)
        # Draw two horizontal lines dividing the frame into three parts
             #cv2.line(image, (0, image.shape[0] // 3), (image.shape[1], image.shape[0] // 3), (0, 255, 0), 2)
             #cv2.line(image, (0, 2 * image.shape[0] // 3), (image.shape[1], 2 * image.shape[0] // 3), (0, 255, 0), 2)  
            if displacement < threshold:
                fingers = detectorHand.fingersUp(hand)
            else:
    # Hand is moving
                fingers = detectorHand.fingersUp(hand)
                if fingers==[1,1,1,1,1]:
                    if cx < image.shape[1] // 2:
                        # Hand is on the left side of the screen
                            if prev_position != 'left':
                                cv2.putText(image, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                                print("Left")
                                image_index = max(image_index-2, 0)
                                prev_position = 'left'
                    elif cx > image.shape[1] // 2:
                        
                            # Hand is on the right side of the screen
                            if prev_position != 'right':
                                cv2.putText(image, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
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
                    
                    if cy > 2 * image.shape[0] // 3:
                        # Hand is below the lower line
                        if prev_position != 'down':
                            cv2.putText(image, 'Down', org, font, fontScale, color, thickness, cv2.LINE_AA)
                            print("Down")
                            #Move image down by 10 pixels
                            #y_offset += 10
                            prev_position = 'down'
                    elif cy < image.shape[0] // 3:
                        # Hand is above the upper line
                        if prev_position != 'up':
                            cv2.putText(image, 'Up', org, font, fontScale, color, thickness, cv2.LINE_AA)
                            print("Up")
                            # Move image up by 10 pixels
                            #y_offset -= 10
                            prev_position = 'up'
                prev_cx, prev_cy = cx, cy
            
    cv2.flip(image,1)
    cv2.imshow("Slides", imageCurrent)
    cv2.imshow("Image", image) 
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






