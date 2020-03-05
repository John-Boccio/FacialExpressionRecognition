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
from PIL import Image
from utils import DatasetType
from utils import FerUtils
import ConfigParser as Cp
import csv
import numpy as np
import os
import pickle
import torch
import warnings


class FER2013Dataset(Dataset):
    def __init__(self, ferplus=True, set_type=DatasetType.TRAIN, tf=None):
        self.ferplus = ferplus
        self.transform = tf
        self.set_type = set_type

        # If dataset is not initialized, check if we have it pickled
        if not ferplus and os.path.exists("./metadata/fer2013/fer2013.pickle"):
            fer2013 = pickle.load(open("./metadata/fer2013/fer2013.pickle", "rb"))
            self.train = fer2013["train"]
            self.test = fer2013["test"]
            self.validation = fer2013["validation"]
            return
        elif ferplus and os.path.exists("./metadata/fer2013/fer2013plus.pickle"):
            fer2013 = pickle.load(open("./metadata/fer2013/fer2013plus.pickle", "rb"))
            self.train = fer2013["train"]
            self.test = fer2013["test"]
            self.validation = fer2013["validation"]
            return

        # Initialize it the hard way
        self.train = []
        self.test = []
        self.validation = []

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
                exp = int(entry[0])
                pixels = entry[1]

                # Convert string of pixel values to image
                pixels = [int(pixel) for pixel in pixels.split(' ')]
                pixels = np.asarray(pixels).reshape((48, 48))
                # pixels = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)

                # Create data point
                data_point = {
                    "img": np.asarray(pixels).astype('uint8'),
                    # The expression labels follow the Expression enum exactly
                    "expression": exp
                }

                # If we are using FER+, we need to change the expression based on new labeling
                if ferplus:
                    plus_entry = next(ferplus_reader)
                    exp_distribution = [int(e) for e in plus_entry[2:]]
                    exp = np.argmax(exp_distribution)
                    if exp == 9:    # Not labeled identifier
                        continue
                    data_point["expression"] = exp

                if entry[2] == "Training":
                    self.train.append(data_point)
                elif entry[2] == "PublicTest":
                    self.validation.append(data_point)
                elif entry[2] == "PrivateTest":
                    self.test.append(data_point)
                else:
                    warnings.warn("Unknown dataset type: " + entry[2])

        dump = {"train":        self.train,
                "validation":   self.validation,
                "test":         self.test}
        if ferplus:
            ferplus_csv.close()
            pickle.dump(dump, open("./metadata/fer2013/fer2013plus.pickle", "wb"))
        else:
            pickle.dump(dump, open("./metadata/fer2013/fer2013.pickle", "wb"))

    def __len__(self):
        if self.set_type == DatasetType.TRAIN:
            return len(self.train)
        elif self.set_type == DatasetType.TEST:
            return len(self.test)
        elif self.set_type == DatasetType.VALIDATION:
            return len(self.validation)
        return -1

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Deep copies so the user can't mess with the dataset
        if self.set_type == DatasetType.TRAIN:
            sample = self.train[item].copy()
        elif self.set_type == DatasetType.VALIDATION:
            sample = self.validation[item].copy()
        elif self.set_type == DatasetType.TEST:
            sample = self.test[item].copy()
        else:
            return None

        sample["img"] = Image.fromarray(sample["img"])
        if self.transform:
            sample["img"] = self.transform(sample["img"])
        return sample
