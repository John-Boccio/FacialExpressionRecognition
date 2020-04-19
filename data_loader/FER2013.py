"""
Author(s):
    John Boccio
Last revision:
    10/18/2019
Description:
    API for using the "FER2013" dataset. To use this, you must have the paths defined in the config.json. This API
    splits the data into a training set (data labeled "Training") and a test set (data labeled "PublicTest" or
    "PrivateTest").
    label：(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
"""
from torch.utils.data import Dataset
from utils import DatasetType
from image_processing import crop_faces
from PIL import Image
import ConfigParser as Cp
import csv
import cv2
import numpy as np
import torch


class FER2013Dataset(Dataset):
    def __init__(self, ferplus=True, facecrop=False, set_type=DatasetType.TRAIN, tf=None):
        self.ferplus = ferplus
        self.transform = tf
        self.set_type = set_type

        self.dataset = []

        if ferplus:
            fer2013plus_config = Cp.ConfigParser.get_config()["data_loader"]["FERplus"]
            ferplus_csv = open(fer2013plus_config["csv_path"])
            # Discard first line due to column labeling
            ferplus_csv.readline()
            ferplus_reader = csv.reader(ferplus_csv, delimiter=',')

        fer2013_config = Cp.ConfigParser.get_config()["data_loader"]["FER2013"]
        # Read CSV containing images, labels, and set type
        with open(fer2013_config["csv_path"]) as csv_file:
            # Remove first row since it just says the ordering of data
            csv_file.readline()

            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, entry in enumerate(csv_reader):
                if ferplus:
                    plus_entry = next(ferplus_reader)

                if self.set_type == DatasetType.TRAIN and entry[2] != "Training":
                    continue
                elif self.set_type == DatasetType.VALIDATION and entry[2] != "PublicTest":
                    continue
                elif self.set_type == DatasetType.TEST and entry[2] != "PrivateTest":
                    continue

                if not ferplus:
                    exp = int(entry[0])
                else:
                    exp_distribution = [int(e) for e in plus_entry[2:]]
                    exp = np.argmax(exp_distribution)
                    if exp == 9:    # Not labeled identifier
                        continue
                pixels = entry[1]

                # Convert string of pixel values to image
                pixels = [int(pixel) for pixel in pixels.split(' ')]
                pixels = np.asarray(pixels).reshape((48, 48)).astype('uint8')

                if facecrop:
                    # Need to convert into 3 channel image
                    pixels = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)
                    faces = crop_faces(pixels)
                    if len(faces) != 1:
                        continue
                    pixels = cv2.cvtColor(faces[0]['img'], cv2.COLOR_RGB2GRAY)

                pixels = Image.fromarray(pixels)
                if tf:
                    pixels = tf(pixels)

                # Create data point
                data_point = {
                    "img": pixels,
                    # The expression labels follow the Expression enum exactly
                    "expression": exp
                }

                self.dataset.append(data_point)

        if ferplus:
            ferplus_csv.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        return self.dataset[item]
