"""
Author(s):
    Muhammad Hussain & John Boccio
Last revision:
    11/22/2019
Description:
   Provides image processing functions.
"""
from PIL import Image
import cv2
import numpy


def crop_faces(image):
    face_data = "image_processing/resources/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(face_data)

    opencv_img = cv2.cvtColor(numpy.asarray(image).astype('uint8'), cv2.COLOR_RGB2BGR)
    minisize = (opencv_img.shape[1], opencv_img.shape[0])
    miniframe = cv2.resize(opencv_img, minisize)

    faces = cascade.detectMultiScale(miniframe)
    crops = []
    for f in faces:
        x, y, w, h = [v for v in f]
        sub_face = opencv_img[y:y+h, x:x+h]
        sub_face = Image.fromarray(cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB))
        crop = {
            "coord": (x, y),
            "size": (w, h),
            "img": sub_face
        }
        crops.append(crop)

    return crops


def crop_face_transform(image):
    faces = crop_faces(image)
    if len(faces) == 0:
        return image
    return faces[0]["img"]

