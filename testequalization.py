import cv2
import numpy as np
import os
DIR = 'C:/Users/Owner/PycharmProjects/untitled2/output/'
pictures = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]

for pic in pictures:
    img = cv2.imread(DIR + str(pic))

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    face_file_name = "C:/Users/Owner/PycharmProjects/untitled2/normalized/" + str(pic) + ".jpg"
    cv2.imwrite(face_file_name, img_output)
    cv2.imshow('Color input image', img)
    cv2.imshow('Histogram equalized', img_output)

cv2.waitKey(0)