"""
Author(s):
    Muhammad Hussain & John Boccio
Last revision:
    11/22/2019
Description:
   Provides image processing functions.
"""
# Importing Modules
from __future__ import print_function
import numpy as np
import cv2

# Face Crop Function using HaarCascade and OpenCV
def crop_faces(image, fx=1.0, fy=1.0):
    # Create a Haar-like feature cascade classifier object
    face_data = "./image_processing/resources/haarcascade_frontalface_default.xml"
    # the classifier that will be used in the cascade
    cascade = cv2.CascadeClassifier(face_data)
    # convert image to an array object
    opencv_img = np.array(image)
    # resize focus based on area of frame and place in miniframe
    small_image = cv2.resize(opencv_img, (0, 0), fx=fx, fy=fy)
    # begin face cascade
    faces = cascade.detectMultiScale(small_image)
    crops = []
    # detect the coordinates of each vertex and the size of rectangle for face
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
    # return cropped image
    return crops

# Alternative Face Crop Method for comparison using face_recognition library
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

# Histogram Equalization Function
def histogram_equalization(image):
    img = np.array(image)
    # convert between RGB/BGR and YUV
    # encodes a color image or video taking human perception into account,
    # allowing reduced bandwidth for chrominance components, thereby
    # enabling transmission errors or compression artifacts to be more
    # efficiently masked by human perception than using a "direct" RGB-representation.
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to BGR format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

# Gamma Correction Function
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


def crop_face_transform(image, fx=1.0, fy=1.0):
    # perform face crop
    faces = crop_faces(image, fx=fx, fy=fy)
    if len(faces) == 0:
        # return image, coordinates and size of crop.
        return {'img': image, 'coord': (0, 0), 'size': (image.shape[0], image.shape[1])}
    return faces[0]
