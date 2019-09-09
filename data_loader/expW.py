from torch.utils.data import Dataset
from skimage import io, transform
import torch
import os


class ExpWDataset(Dataset):
    def __init__(self, image_dir, label_path, tf=None):
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = tf

        """
        
        label.lst： each line indicates an image as follows:
        image_name face_id_in_image face_box_top face_box_left face_box_right face_box_bottom face_box_cofidence expression_label

        for expression label：
        "0" "angry"
        "1" "disgust"
        "2" "fear"
        "3" "happy"
        "4" "sad"
        "5" "surprise"
        "6" "neutral"
        """
        # Dictionary which contains the labeled data for each of the images in the dataset
        self.images = {}
        with open(self.label_path) as labels:
            line = labels.readline()
            while line:
                label = line.split(' ')
                img_name = label[0]
                file_path = os.path.join(self.image_dir, img_name)
                if not os.path.isfile(file_path):
                    print("WARNING: Label for {} present in {} but is not in {}, skipping..."
                          .format(img_name, self.label_path, self.image_dir))
                else:
                    # Add the new image to the image dictionary
                    self.images.update(
                        {
                            file_path: {
                                label[1]: {
                                    "face_box": {
                                        "top":        int(label[2]),
                                        "left":       int(label[3]),
                                        "right":      int(label[4]),
                                        "bottom":     int(label[5]),
                                        "confidence": float(label[6])
                                    },
                                    "expression": label[7]
                                }
                            }
                        }
                    )

                # Read the next label
                line = labels.readline()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = list(self.images.keys())[idx]
        image = io.imread(image_path)
        sample = {
            "name": image_path,
            "image": image,
            "label": self.images[image_path]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
