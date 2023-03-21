import cv2 
import mediapipe as mp
import numpy as np
import UdpComms as U
from HandTrackingModule import HandDetector

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the Mediapipe Holistic model
mp_holistic = mp.solutions.holistic

# Initialize the drawing utility
mp_drawing = mp.solutions.drawing_utils


sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

camera.set(3, 500)
camera.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        # Detect hands in the frame
        hands, frame = detector.findHands(frame)
        
        # Get the frame width and height
        height, width, _ = frame.shape
        
        # Draw two vertical lines to divide the screen into three equal parts
        line_width = 2
        line_color = (0, 0, 255)
        x1 = width // 3
        x2 = 2 * width // 3
        cv2.line(frame, (x1, 0), (x1, height), line_color, line_width)
        cv2.line(frame, (x2, 0), (x2, height), line_color, line_width)
        
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect the body landmarks
        results = holistic.process(frame)
        
        # Compute the centroid coordinates
        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            coordinates = np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmarks])
            centroid = np.mean(coordinates, axis=0)
        
            # Draw the centroid on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.circle(frame, (int(centroid[0]*frame.shape[1]), int(centroid[1]*frame.shape[0])), 5, (0, 255, 0), -1)
            x = f"{(centroid[0]*10)-5:.2f}"
            y = f"{(centroid[1]*10-5):.2f}"
            z = 0
            Body = (x, y)
        else:
            Body = []

        # Combine body and hands to create the message
        if hands:
            # Hand 1
            hand = hands[0]
            lmList = hand["lmList"]  # List of 21 Landmark points
            data = []
            for lm in lmList:
                data.extend([lm[0], height - lm[1], lm[2]])
            Hands = data
        else:
            Hands = []
        message = {"Body": Body , "Hands": Hands}
        print(message)
        # Send the message over UDP
        sock.SendData(f"{message}")
        
        # Display the frame
        cv2.imshow('frame', cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_RGB2BGR))
        
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and destroy all windows
camera.release()
cv2.destroyAllWindows()
