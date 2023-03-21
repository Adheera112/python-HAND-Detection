# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:56:26 2023

@author: HP
"""

import cv2 
import imutils 
   
# Initializing the HOG person 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   
# Reading the Image 
image = cv2.imread("C:/Users/HP/Desktop/internship/f011a11fc4263f34c63e3a72c2f9643e.jpg") 
   
# Resizing the Image 
image = imutils.resize(image,width=min(500, image.shape[1])) 
   
# Detecting all humans 
(humans, _) = hog.detectMultiScale(image,  
                                    winStride=(5, 5), 
                                    padding=(3, 3), 
                                    scale=1.21)
# getting no. of human detected
print('Human Detected : ', len(humans))
   
# Drawing the rectangle regions
for (x, y, w, h) in humans: 
    cv2.rectangle(image, (x, y),  
                  (x + w, y + h),  
                  (0, 0, 255), 2) 
  
# Displaying the output Image 
cv2.imshow("Image", image) 
cv2.waitKey(0) 
   
cv2.destroyAllWindows() 