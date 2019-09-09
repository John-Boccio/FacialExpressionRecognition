from ConfigParser import *
from data_loader import expW
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_expw_image(ax, image, face_box, expression):
    plt.imshow(image)
    width = face_box["right"] - face_box["left"]
    height = face_box["bottom"] - face_box["top"]
    box = patches.Rectangle((face_box["left"], face_box["top"]), width, height,
                            linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(box)


config = ConfigParser.get_config()

expw_config = ConfigParser.get_config()["data_loader"]["expW"]
expw_dataset = expW.ExpWDataset(image_dir=expw_config["image_dir"],
                                label_path=expw_config["label_path"])

for i in range(len(expw_dataset)):
    sample = expw_dataset[i]
    sample_label = sample["label"]
    for face_id in sample["label"]:
        fig, ax = plt.subplots(1)
        face_label = sample_label[face_id]
        show_expw_image(ax, sample["image"], face_label["face_box"], face_label["expression"])
        plt.show()
