import cv2
import mediapipe as mp

# Initialize Mediapipe hand detection model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize variables for hand landmarks and line drawing
prev_landmarks = None
curr_landmarks = None
line_start = None
line_end = None

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Flip the frame horizontally to mimic a mirror
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame using Mediapipe hand detection model
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Convert the RGB frame to a Mediapipe image
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Process the image and detect hand landmarks
        results = hands.process(image)

        # Get the landmarks of the detected hand
        if results.multi_hand_landmarks:
            # Get the landmarks of the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Convert the hand landmarks to pixel coordinates
            h, w, c = frame.shape
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))

            # Update the previous and current landmarks
            prev_landmarks = curr_landmarks
            curr_landmarks = landmarks

            # Draw the landmarks on the frame
            # Draw the landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))


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
                if abs(dx) > abs(dy) and dx > 0:
                    direction = "right"
                elif abs(dx) > abs(dy) and dx < 0:
                    direction = "left"
                elif abs(dy) > abs(dx) and dy > 0:
                    direction = "down"
                elif abs(dy) > abs(dx) and dy < 0:
                    direction = "up"

                # Draw the line on the frame and the direction of hand movement
                cv2.line(frame, line_start, line_end, (0, 0, 255), thickness=2)
                cv2.putText(frame, direction, (line_end[0]+10, line_end[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    # Display the frame
    cv2.imshow('Hand Tracking',frame)
    if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
