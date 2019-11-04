import FacialExpressionRecognition.ConfigParser as Cp
import FacialExpressionRecognition.neural_nets as nns
import torchvision.transforms as transforms


main = Cp.ConfigParser.get_config()["main"]
to_run = main["run"]
task = main["task"]

if to_run["dataset"] == "fer2013":
    net = None
    transform = None
    if to_run["model"] == "Fer2013V1":
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        net = nns.Fer2013V1(tf=transform)
    else:
        print("Invalid config.json file for main")
        exit()

    nns.fer2013_train_nn(net, "metadata/neural_nets/Fer2013V1.torch")
