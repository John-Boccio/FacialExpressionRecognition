from __future__ import print_function

import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('../PycharmProjects/Untitled2/test/assets/aflw-test.jpg')
preds = fa.get_landmarks(input)

