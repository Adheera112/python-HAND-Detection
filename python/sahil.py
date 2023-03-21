import cv2 
import mediapipe as mp
import numpy as np
import UdpComms as U
from HandTrackingModule import HandDetector
import math

class PoseDetector:
    def __init__(self, udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True, detectionCon=0.5, trackingCon=0.5, maxHands=1):
        self.camera = cv2.VideoCapture(0)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.sock = U.UdpComms(udpIP=udpIP, portTX=portTX, portRX=portRX, enableRX=enableRX, suppressWarnings=suppressWarnings)
        self.camera.set(3, 500)
        self.camera.set(4, 720)
        self.detector = HandDetector(detectionCon=detectionCon, maxHands=maxHands)
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=detectionCon, min_tracking_confidence=trackingCon)
        self.results = None

    def __del__(self):
        self.camera.release()

    def get_pose(self):
        ret, frame = self.camera.read()
        # Detect hands in the frame
        hands, frame = self.detector.findHands(frame)
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
        self.results = self.holistic.process(frame)
        # Compute the centroid coordinates
        if self.results.pose_landmarks is not None:
            landmarks = self.results.pose_landmarks.landmark
            right_shoulder_landmark = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_SHOULDER]
            right_hip_landmark = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP]
            right_knee_landmark = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE]
            right_ankle_landmark = self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE]
            x1, y1, z1 = right_shoulder_landmark.x, right_shoulder_landmark.y, right_shoulder_landmark.z
            x2, y2, z2 = right_hip_landmark.x, right_hip_landmark.y, right_hip_landmark.z
            x3, y3, z3 = right_knee_landmark.x, right_knee_landmark.y, right_knee_landmark.z
            x4, y4, z4 = right_ankle_landmark.x, right_ankle_landmark.y, right_ankle_landmark.z

            DRS2DRH = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            DRH2DRK = math.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)
            DRK2DRA = math.sqrt((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2)

            # print(DRS2DRH)

            length = f"{(DRS2DRH + DRH2DRK + DRK2DRA)*10:.2f}"
            # print("height:",height)

            #getting centroid for all landmarks.........

            coordinates = np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmarks])
            centroid = np.mean(coordinates, axis=0)
            # Draw the centroid on the frame
            self.mp_drawing.draw_landmarks(frame,  self.results.pose_landmarks,  self.mp_holistic.POSE_CONNECTIONS)
            cv2.circle(frame, (int(centroid[0]*frame.shape[1]), int(centroid[1]*frame.shape[0])), 5, (0, 255, 0), -1)
            x = f'{(centroid[0]*10)-5:.2f}'
            y = f'{(centroid[1]*10-5):.2f}'
            z = 0
            Body = x+","+y
            # print(Body)
        else:
            Body = []
        # Display the frame
        cv2.imshow('frame', cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_RGB2BGR))
        
        