"""
Author(s):
    John Boccio
Last revision:
    10/18/2019
Description:

"""
import data_loader as dl
import matplotlib.pyplot as plt

ck_train = dl.CKDataset(train=True)
fer_train = dl.FER2013Dataset(train=True)
expw_train = dl.ExpWDataset(train=True)

for i in range(len(expw_train)):
    sample = expw_train[i]
    imgplot = plt.imshow(sample["img"], cmap='gray')
    plt.title(dl.Expression(sample["expression"]))
    plt.show()
    plt.clf()
