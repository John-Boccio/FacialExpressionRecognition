"""
Author(s):
    John Boccio
Last revision:
    10/18/2019
Description:
    API for using the "Expressions in the Wild" dataset. To use this, you must have the paths defined in the
    config.json.
    Labels:
        "0" "angry"
        "1" "disgust"
        "2" "fear"
        "3" "happy"
        "4" "sad"
        "5" "surprise"
        "6" "neutral"
    https://cs.anu.edu.au/few/AFEW.html
"""
from torch.utils.data import Dataset
from PIL import Image
from utils import DatasetType
import ConfigParser as Cp
import torch
import pickle
import numpy
import os
import warnings


class ExpWDataset(Dataset):
    __train = None
    __test = None
    __validation = None

    def __init__(self, set_type=DatasetType.TRAIN, tf=None):
        self.set_type = set_type
        self.transform = tf

        if ExpWDataset.__train is not None \
                and ExpWDataset.__test is not None \
                and ExpWDataset.__validation is not None:
            return

        # If dataset is not initialized, check if we have it pickled
        if os.path.exists("./metadata/expw/expw.pickle"):
            dataset = pickle.load(open("./metadata/expw/expw.pickle", "rb"))
            ExpWDataset.__train = dataset['train']
            ExpWDataset.__test = dataset['test']
            ExpWDataset.__validation = dataset['validation']
            return

        expw_config = Cp.ConfigParser.get_config()["data_loader"]["expW"]
        image_dir = expw_config["image_dir"]
        label_path = expw_config["label_path"]
        """
        label.lst： each line indicates an image as follows:
        image_name face_id_in_image face_box_top face_box_left face_box_right face_box_bottom face_box_confidence expression_label
        """
        # Dictionary which contains the labeled data for each of the images in the dataset
        data = []
        with open(label_path) as labels:
            line = labels.readline()
            while line:
                label = line.split(' ')
                img_name = label[0]
                file_path = os.path.join(image_dir, img_name)
                if not os.path.isfile(file_path):
                    warnings.warn("WARNING: Label for {} present in {} but is not in {}, skipping...".format(img_name,
                                                                                                             label_path,
                                                                                                             image_dir))
                else:
                    # Add the new image to the image dictionary
                    data_point = {
                        "img_path": file_path,
                        "face_box": {
                            "top": int(label[2]),
                            "left": int(label[3]),
                            "right": int(label[4]),
                            "bottom": int(label[5]),
                            "confidence": float(label[6])
                        },
                        # The emotion labels follow the Expression enum exactly
                        "expression": int(label[7])
                    }
                    data.append(data_point)

                # Read the next label
                line = labels.readline()

        # TODO: Go through the dictionary with our own face detector to validate that the labeled boxes match up with
        #   faces. If they don't, we should delete that image from the dataset.

        # Split into test and training sets
        # TODO: Do this in a smarter way by making sure the test set has an even distribution of expressions
        ExpWDataset.__train = data[:int(len(data) * .8)]
        ExpWDataset.__test = data[int(len(data) * .8):int(len(data) * .95)]
        ExpWDataset.__validation = data[int(len(data) * .95):]

        dataset = {'train': ExpWDataset.__train,
                   'test': ExpWDataset.__test,
                   'validation': ExpWDataset.__validation}
        pickle.dump(dataset, open("./metadata/expw/expw.pickle", "wb"))

    def __len__(self):
        if self.set_type == DatasetType.TRAIN:
            return len(ExpWDataset.__train)
        elif self.set_type == DatasetType.TEST:
            return len(ExpWDataset.__test)
        elif self.set_type == DatasetType.VALIDATION:
            return len(ExpWDataset.__validation)
        return -1

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Deep copies so the user can't mess with the dataset
        if self.set_type == DatasetType.TRAIN:
            data = ExpWDataset.__train[item].copy()
        elif self.set_type == DatasetType.TEST:
            data = ExpWDataset.__test[item].copy()
        elif self.set_type == DatasetType.VALIDATION:
            data = ExpWDataset.__validation[item].copy()
        else:
            return None

        image = Image.open(data["img_path"])
        sample = {
            "img": image,
            "face_box": data["face_box"],
            "expression": data["expression"]
        }

        if self.transform:
            sample["img"] = self.transform(sample["img"])
        return sample
