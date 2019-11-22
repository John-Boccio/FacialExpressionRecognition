"""
Author(s):
    John Boccio
Last revision:
    10/18/2019
Description:
    API for using the "CK+" dataset. To use this, you must have the paths defined in the config.json. This API only uses
    the "peak" emotion pictures from the dataset asa those are the ones that are labeled with facial expressions.
    Labels: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
    Since this data set has an extra label (contempt), those data points will be removed

    http://www.consortium.ri.cmu.edu/ckagree/
"""
from torch.utils.data import Dataset
from skimage import io
from utils import DatasetType
from utils import Expression
import ConfigParser as Cp
import os
import pickle
import torch


class CKDataset(Dataset):
    __train = None
    __test = None
    __validation = None

    def __init__(self, set_type=DatasetType.TRAIN, tf=None):
        self.transform = tf
        self.set_type = set_type

        # Check if the dataset has already been initialized
        if CKDataset.__train is not None \
                and CKDataset.__test is not None \
                and CKDataset.__validation is not None:
            return

        # If dataset is not initialized, check if we have it pickled
        if os.path.exists("./metadata/ck/ck.pickle"):
            dataset = pickle.load(open("./metadata/ck/ck.pickle", "rb"))
            CKDataset.__train = dataset['train']
            CKDataset.__test = dataset['test']
            CKDataset.__validation = dataset['validation']
            return

        # Initialize it the hard way
        data = []

        ck_config = Cp.ConfigParser.get_config()["data_loader"]["CK"]
        # Go through the image directory and find images for the associated dataset type
        img_dir = ck_config["image_dir"]
        emotion_dir = ck_config["emotion_dir"]
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                img_path = os.path.join(root, file)
                rel_img_path = img_path.replace(img_dir, '')

                # Find the associated emotion
                # Get the full path for the emotion file for this image
                emotion_path = emotion_dir + rel_img_path
                # The emotion file name is the same as the image file name with _emotion.txt instead of .png
                emotion_path = emotion_path.replace(".png", "_emotion.txt")

                # The emotion label only exists for peak emotions in the sequences
                if os.path.isfile(emotion_path):
                    with open(emotion_path) as f:
                        emotion = int(f.readline().strip(' ')[0])
                    emotion = CKDataset.ck_to_expression(emotion)
                    if emotion is None:
                        continue

                    data_point = {
                        "img_path": img_path,
                        # Adjust the expression to match the Expression Enum
                        "expression": emotion
                    }
                    data.append(data_point)

        # Split into train and test
        CKDataset.__train = data[:int(len(data)*.80)]
        CKDataset.__test = data[int(len(data)*.80):int(len(data)*.95)]
        CKDataset.__validation = data[int(len(data)*.95):]

        dataset = {'train': CKDataset.__train,
                   'test': CKDataset.__test,
                   'validation': CKDataset.__validation}
        pickle.dump(dataset, open("./metadata/ck/ck.pickle", "wb"))

    def __len__(self):
        if self.set_type == DatasetType.TRAIN:
            return len(CKDataset.__train)
        elif self.set_type == DatasetType.TEST:
            return len(CKDataset.__test)
        elif self.set_type == DatasetType.VALIDATION:
            return len(CKDataset.__validation)
        return -1

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Deep copies so the user can't mess with the dataset
        if self.set_type == DatasetType.TRAIN:
            data = CKDataset.__train[item].copy()
        elif self.set_type == DatasetType.TEST:
            data = CKDataset.__test[item].copy()
        elif self.set_type == DatasetType.VALIDATION:
            data = CKDataset.__validation[item].copy()
        else:
            return None

        sample = {
            "img": io.imread(data["img_path"]),
            "expression": data["expression"]
        }

        if self.transform:
            sample = self.transform(sample["img"])

        return sample

    @staticmethod
    def ck_to_expression(expression):
        if expression == 0:
            return Expression.NEUTRAL.value
        elif expression == 1:
            return Expression.ANGRY.value
        elif expression == 3:
            return Expression.DISGUST.value
        elif expression == 4:
            return Expression.FEAR.value
        elif expression == 5:
            return Expression.HAPPY.value
        elif expression == 6:
            return Expression.SAD.value
        elif expression == 7:
            return Expression.SURPRISE.value
        return None

