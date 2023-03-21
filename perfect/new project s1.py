import cv2
import mediapipe as mp
screen_height =200
screen_width = 200
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

prev_x, prev_y = None, None
# Initialize variables for hand landmarks and line drawing
prev_landmarks = None
curr_landmarks = None
line_start = None
line_end = None

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
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

            # Draw the landmarks on the image
            # Draw the landmarks on the image
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))


            # Draw a line based on the movement of the landmarks
            if prev_landmarks is not None and curr_landmarks is not None and prev_landmarks[8] != curr_landmarks[8]:
                prev_x, prev_y = prev_landmarks[8]  # index finger tip
                curr_x, curr_y = curr_landmarks[8]  # index finger tip
                if line_start is None or prev_x < 0 or prev_x > screen_width or prev_y < 0 or prev_y > screen_height:
                    line_start = None  # Start a new line if hand goes out of screen
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


                # Draw the line on the image and the direction of hand movement
                cv2.line(image, line_start, line_end, (0, 0, 255), thickness=2)
                cv2.putText(image, direction, (line_end[0]+10, line_end[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    # Display the image
    cv2.imshow('Hand Tracking',image)
    if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

