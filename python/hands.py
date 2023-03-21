import cv2
import mediapipe as mp
import numpy as np
import UdpComms as U

class HandDetection:
    def __init__(self, udpIP="127.0.0.1", portTX=8000, portRX=8001):
        self.hand_detected = False
        self.x, self.y, self.z = 0, 0, 0  # initial position
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.sock = U.UdpComms(udpIP=udpIP, portTX=portTX, portRX=portRX, enableRX=True, suppressWarnings=True)
        self.buffer_size = 10
        self.threshold = 1
        self.prev_x = None
        self.x_buffer = []

    def detect(self):
        with self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                results = hands.process(frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if not self.hand_detected:
                            self.x, self.y, self.z = 0, 0, 0
                            self.hand_detected = True
                        else:
                            self.x = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1]
                            self.y = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0]
                            self.z = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
                            self.x = round(self.x / 100, 0)
                            self.y = round(self.y / 100, 0)
                            self.z = round(self.z / 100, 0)
                            self.sock.SendData(f"{self.x},{self.y}")
                            print(self.x, self.y, self.z)  # added print statement
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                else:
                    self.x, self.y, self.z = 0, 0, 0
                    self.sock.SendData(f"{self.x},{self.y}")
                self.x_buffer.append(self.x)
                if len(self.x_buffer) > self.buffer_size:
                    self.x_buffer.pop(0)
                if len(self.x_buffer) == self.buffer_size:
                    x_avg = np.mean(self.x_buffer)
                    if self.prev_x is not None and abs(x_avg - self.prev_x) < self.threshold:
                        if x_avg < self.prev_x:
                            self.sock.SendData("left")
                        elif x_avg > self.prev_x:
                            self.sock.SendData("right")
                    self.prev_x = x_avg
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Hand Detection', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows()
