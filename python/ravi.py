from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import dlib
import numpy as np
from math import hypot
import csv

class FacialLandmarksDetector:
    def __init__(self):
        self.detector_face = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("C:/Users/ravip/summer internship/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
        self.detector = FaceMeshDetector(maxFaces=1)
        self.distance = None
        self.pos = []

    def get_eye_distances(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector_face(gray)

        if len(faces) == 1:
            for face in faces:
                landmarks = self.predictor(gray, face)
                left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                left_eye_center = np.mean(left_eye_landmarks, axis=0).astype(np.int64)
                right_eye_center = np.mean(right_eye_landmarks, axis=0).astype(np.int64)
                left_eye = frame[left_eye_center[1] - 50:left_eye_center[1] + 50,left_eye_center[0] - 50:left_eye_center[0] + 50]
                right_eye = frame[right_eye_center[1] - 50:right_eye_center[1] + 50,right_eye_center[0] - 50:right_eye_center[0] + 50]
                self.distance = get_distance(left_eye_center, right_eye_center)/10
                cv2.circle(frame,tuple(left_eye_center),2,(0,0,255), -1)
                cv2.circle(frame,tuple(right_eye_center),2,(0,0,255), -1)
                cv2.putText(frame, f"eyes distance: {self.distance:.1f}cm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def get_face_distance(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector_face(gray)

        if len(faces) == 0:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            width_in_pixels,_ = self.detector.findDistance(pointLeft,pointRight)
            Width_in_cm = 6.3
            focal_length = 800
            d = (Width_in_cm*focal_length)/width_in_pixels
            cv2.putText(frame,f'Distance between cam to face: {int(d):.1f}cm.', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("iris distance", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = FacialLandmarksDetector()
    while True:
        success, frame = cap.read()
        frame, faces = detector.get_face_distance(frame=frame)
        if faces:
            print(faces[0])
        cv2.imshow("Image", frame)
        cv2.waitKey(1)


main()