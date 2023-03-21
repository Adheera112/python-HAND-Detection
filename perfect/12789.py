# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:20:40 2023

@author: 001ku
"""

import cv2
import numpy as np

# Load the input image
img = cv2.imread('input_image.jpg')

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the input image
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

# Loop through all detected faces and remove the skin region
for (x, y, w, h) in faces:
    # Extract the face region
    face = img[y:y+h, x:x+w]
    
    # Convert the face to the HSV color space
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds of the skin color in HSV
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    # Create a binary mask of the skin region
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply the mask to the face region
    face[mask != 0] = [255, 255, 255]
    
    # Replace the original face region with the modified one
    img[y:y+h, x:x+w] = face

# Save the output image
cv2.imwrite('output_image.jpg', img)
