"""
Author(s):
    Muhammad Hussain & John Boccio
Last revision:
    11/22/2019
Description:
   Provides image processing functions.
"""
from __future__ import print_function
import numpy as np
import argparse
import cv2


def crop_faces(image):
    face_data = "./image_processing/resources/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(face_data)

    opencv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    minisize = (opencv_img.shape[1], opencv_img.shape[0])
    miniframe = cv2.resize(opencv_img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    crops = []
    for f in faces:
        x, y, w, h = [v for v in f]
        sub_face = opencv_img[y:y + h, x:x + h]
        sub_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB)
        crop = {
            "coord": (x, y),
            "size": (w, h),
            "img": sub_face
        }
        crops.append(crop)

    return crops


def histogram_equalization(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

    return img_output


def adjust_gamma(image, gamma=2.0):
    image = np.array(image)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    gc = cv2.LUT(image, table)
    return gc


def crop_face_transform(image):
    faces = crop_faces(image)
    faces = adjust_gamma(faces, gamma=2.0)
    if len(faces) == 0:
        return image
    return faces[0]["img"]
