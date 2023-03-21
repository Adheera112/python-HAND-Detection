import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#image path
folderPath = "Presentation"
image_index = 0
# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
resizedImages = [cv2.resize(cv2.imread(os.path.join(folderPath, imagePath)), (720, 420)) for imagePath in pathImages]

# Variables
org = (40, 40)
fontScale = 1
color = (255, 0, 0)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX


# Initialize variables for hand landmarks and line drawing
prev_landmarks = None
curr_landmarks = None
line_start = None
line_end = None
prev_x, prev_y = None, None
# Initialize prev_distance_to_center
prev_distance_to_center = float('inf')
prev_midpoint = None

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    pathFullImage = os.path.join(folderPath, pathImages[image_index])
    imageCurrent = cv2.imread(pathFullImage)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert the hand landmarks to pixel coordinates
            h, w, c = image.shape
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))

            # Update the previous and current landmarks
            prev_landmarks = curr_landmarks
            curr_landmarks = landmarks

            
            # Draw a line based on the movement of the landmarks
            if prev_landmarks is not None and curr_landmarks is not None:
                prev_x, prev_y = prev_landmarks[8]  # index finger tip
                curr_x, curr_y = curr_landmarks[8]  # index finger tip
            
                if line_start is None:
                    line_start = (prev_x, prev_y)
                line_end = (curr_x, curr_y)
            
                # Determine the direction of hand movement based on the line
                dx = line_end[0] - line_start[0]
                dy = line_end[1] - line_start[1]
                direction = ""
                
                # Find the midpoint of the line
                midpoint = ((prev_x + curr_x) // 2, (prev_y + curr_y) // 2)
            
                # Calculate the distance between the midpoint and the center of the screen
                center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
                distance_to_center = ((midpoint[0] - center_x)**2 + (midpoint[1] - center_y)**2)**0.5
            
                if abs(dx) > abs(dy) and dx > 0:
                    direction = "right"
                    # Hand is on the right side of the screen
                    cv2.putText(image, 'Right', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    print("Right")
                    image_index = min(image_index+2, len(pathImages)-1)
                       
                        
                elif abs(dx) > abs(dy) and dx < 0:
                    direction = "left"
                    
                    cv2.putText(image, 'Left', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    print("Left")
                    image_index = max(image_index-2, 0)
                        
                        
                elif abs(dy) > abs(dx) and dy > 0:
                    direction = "down"
                elif abs(dy) > abs(dx) and dy < 0:
                    direction = "up"
                
                # Check if the movement is towards the center of the screen and the hand is moving
                elif prev_midpoint is not None and distance_to_center < prev_distance_to_center and \
                     ((midpoint[0] - prev_midpoint[0])**2 + (midpoint[1] - prev_midpoint[1])**2)**0.5 > 10:
                    direction = "center"
                    
                    cv2.putText(image, 'Center', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    print("Center")
                        
                #Store the current line_start for the next iteration
                line_start = line_end


                # Draw the line on the image and the direction of hand movement
                #cv2.line(image, line_start, line_end, (0, 0, 255), thickness=2)
                cv2.putText(image, direction, (line_end[0]+10, line_end[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Store the current distance and midpoint for the next iteration
                prev_distance_to_center = distance_to_center
                prev_midpoint = midpoint
            else:
                # If there is no movement, set prev_distance_to_center and prev_midpoint to None
                prev_distance_to_center = None
                prev_midpoint = None

                
                
    # Display the image
    cv2.imshow('Hand Tracking',image)
    cv2.imshow('Slide',imageCurrent)
    if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

