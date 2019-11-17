"""
Author(s):
    John Boccio
Last revision:
    10/27/2019
Description:

"""
import torch


def fer2013_train_nn(model, criterion, optimizer, save_path, trainloader, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, sample in enumerate(trainloader):
            inputs = sample['img']
            labels = sample['expression']
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 3000 == 2999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 3000))
                running_loss = 0.0

    torch.save(model.state_dict(), save_path)


def fer2013_test_nn(model, trainloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in trainloader:
            inputs = sample['img']
            labels = sample['expression']
            nn_prediction = model.forward(inputs)
            _, predicted = torch.max(nn_prediction.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy: {} %".format(100*correct/total))

