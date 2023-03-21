import cv2 
import mediapipe as mp
import numpy as np
import UdpComms as U

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the Mediapipe Holistic model
mp_holistic = mp.solutions.holistic

# Initialize the drawing utility
mp_drawing = mp.solutions.drawing_utils

sock = U.UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()


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
            # x = round(x/frame.shape[0]*5-2.5,2)
            y = f"{(centroid[1]*10-5):.2f}"
            z = 0
            print(x,y)
            # Print the centroid coordinates
            # print(f"{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}")
            # sock.SendData(f"{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}")
            sock.SendData(f"{x},{y},{z}")


        # Display the frame
        cv2.imshow('frame', cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_RGB2BGR))

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and destroy all windows
camera.release()
cv2.destroyAllWindows()
