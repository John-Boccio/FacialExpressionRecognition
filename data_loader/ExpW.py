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
from skimage import io, transform
import ConfigParser as Cp
import torch
import pickle
import os


class ExpWDataset(Dataset):
    __train = None
    __test = None

    def __init__(self, train=True, tf=None):
        self.train = train
        self.transform = tf

        if ExpWDataset.__train is not None and ExpWDataset.__test is not None:
            return

        # If dataset is not initialized, check if we have it pickled
        should_pickle = Cp.ConfigParser.get_config()["data_loader"]["pickle"]
        if should_pickle and os.path.exists("./metadata/expw/train.pickle") and \
                os.path.exists("./metadata/expw/test.pickle"):
            ExpWDataset.__train = pickle.load(open("./metadata/expw/train.pickle", "rb"))
            ExpWDataset.__test = pickle.load(open("./metadata/expw/test.pickle", "rb"))
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
                    print("WARNING: Label for {} present in {} but is not in {}, skipping..."
                          .format(img_name, label_path, image_dir))
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
                        # The emotion labels follow the Expression.py enum exactly
                        "expression": int(label[7])
                    }
                    data.append(data_point)

                # Read the next label
                line = labels.readline()

        # TODO: Go through the dictionary with our own face detector to validate that the labeled boxes match up with
        #   faces. If they don't, we should delete that image from the dataset.

        # Split into test and training sets
        # TODO: Do this in a smarter way by making sure the test set has an even distribution of expressions
        ExpWDataset.__train = data[:int(len(data)*.8)]
        ExpWDataset.__test = data[int(len(data)*.8):]

        if should_pickle:
            pickle.dump(ExpWDataset.__train, open("./metadata/expw/train.pickle", "wb"))
            pickle.dump(ExpWDataset.__test, open("./metadata/expw/test.pickle", "wb"))

    def __len__(self):
        if self.train:
            return len(ExpWDataset.__train)
        else:
            return len(ExpWDataset.__test)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self.train:
            data = ExpWDataset.__train[item]
        else:
            data = ExpWDataset.__test[item]

        image = io.imread(data["img_path"])
        sample = {
            "img": image,
            "face_box": data["face_box"],
            "expression": data["expression"]
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
