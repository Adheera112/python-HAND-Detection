import cv2
import UdpComms as U
import mediapipe as mp
import numpy as np
hand_detected = False

x, y, z = 0, 0, 0  # initial position

# Load MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Initialize VideoCapture object
cap = cv2.VideoCapture(0)  # 0 for default camera, or pass the path to a video file
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
# Create a hands object
with mp_hands.Hands(
        max_num_hands=1,  # maximum number of hands to detect
        min_detection_confidence=0.5,  # minimum detection confidence
        min_tracking_confidence=0.5 ) as hands:
    prev_x = None
    x_buffer = []
    buffer_size = 10
    threshold = 1  # threshold for detecting hand movement
    while cap.isOpened():
        # Read the frame from the video stream
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR frame to RGB for Mediapipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)

        # Detect hands in the RGB frame
        results = hands.process(frame)
 
        # Draw the hand landmarks on the RGB frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
               if not hand_detected:
                   x, y, z = 0, 0, 0
                   hand_detected = True
               else:
                   # Get the x, y, and z coordinates of the middle finger tip landmark (Landmark 8)
                   x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1]
                   y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0]
                   z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                   # Round the x, y, and z coordinates to two decimal places
                   x =round(x/100,0)
                   y = round(y/100,0)
                   z = round(z/100,0)
                   
                   # Print the x, y, and z coordinates in pixel coordinates
                   print(f"{x},{y},{z}")
                   # Send hand position coordinates to other application using UDP communication
                   sock.SendData(f"{x},{y}")
                   mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) 

        # If hand is not detected, print x, y, and z as 0
        else:
            x, y, z = 0, 0, 0
            print(" 0,0,0")
            # Send hand position coordinates to other application using UDP communication
        sock.SendData(f"{x},{y}")
        # Add the x-coordinate to the buffer
        x_buffer.append(x)
            # If the buffer is full, remove the oldest value
        if len(x_buffer) > buffer_size:
            x_buffer.pop(0)

            # Compute the moving average of the x-coordinates in the buffer
        if len(x_buffer) == buffer_size:
            x_avg = np.mean(x_buffer)

            # Check for hand movement
            if prev_x is not None and abs(x_avg - prev_x) < threshold: 
                if x_avg < prev_x:
                    x= -2.5
                    #print("left")
                    #sock.SendData("left")
                elif x_avg > prev_x:
                    x= 2.5
                    #print("right")
                    #sock.SendData("right")
                    
            
                    
            
                # Store the current moving average as the previous x-coordinate for the next frame
            prev_x = x_avg
        # Convert the RGB frame back to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
