from handmove import Hand
import cv2

cap = cv2.VideoCapture(0)
hd=Hand()

while True:
    # Get image frame
    success, img = cap.read()
    hd.detect(img)
    cv2.imshow("Image",img)
   
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()