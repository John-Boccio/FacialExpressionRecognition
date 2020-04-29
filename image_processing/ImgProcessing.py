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
import cv2
# import face_recognition
from PIL import Image


def crop_faces(image, fx=1, fy=1):
    face_data = "./image_processing/resources/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(face_data)

    image = cv2.resize(image, (0, 0), fx=fx, fy=fy)
    opencv_img = np.array(image)
    minisize = (opencv_img.shape[1], opencv_img.shape[0])
    miniframe = cv2.resize(opencv_img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    crops = []
    for f in faces:
        x, y, w, h = [v for v in f]
        x *= int(1/fx)
        w *= int(1/fx)
        y *= int(1/fy)
        h *= int(1/fy)
        sub_face = opencv_img[y:y + h, x:x + h]
        crop = {
            "coord": (x, y),
            "size": (w, h),
            "img": sub_face
        }
        crops.append(crop)

    return crops


"""
def face_rec(image):
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

    return pil_image
"""


def histogram_equalization(image):
    img = np.array(image)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to BGR format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


def adjust_gamma(image, gamma=1.25):
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
    if len(faces) == 0:
        return {'img': image, 'coord': (0, 0), 'size': (image.shape[0], image.shape[1])}
    return faces[0]
