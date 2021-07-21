import os
import sys

import PIL.Image
import torch
import torchvision.transforms
import random
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from DataPreparer.DataPreparer import DataPreparer
from DataSetImplementation.CustomImageDataset import CustomImageDataset
from torch import nn
from matplotlib import pyplot as plt
from matplotlib import patches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-3
batch_size = 5
element_count_learn = 3000
element_count_test = 500

data = DataPreparer(
    element_count=element_count_learn,
    data_path='data/first.png',
    out_path='Lerning')

learn_data = CustomImageDataset(
    img_dir='Lerning',
    annotations_file='Lerning/data.csv',
    transform=torchvision.transforms.ToTensor())

data = DataPreparer(
    element_count=element_count_test,
    data_path='data/first.png',
    out_path='Testing')

test_data = CustomImageDataset(
    img_dir='Testing',
    annotations_file='Testing/data.csv',
    transform=torchvision.transforms.ToTensor())

data = DataPreparer(
    element_count=50,
    data_path='data/first.png',
    out_path='Test1')

test1_data = CustomImageDataset(
    img_dir='Test1',
    annotations_file='Test1/data.csv',
    transform=torchvision.transforms.ToTensor())

loader_learn_data = DataLoader(learn_data, batch_size=batch_size, shuffle=True)
loader_test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)
loader_test1_data = DataLoader(test1_data, batch_size=1, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.neural = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=1728, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=4)
        )

    def forward(self, x):
        return self.neural(x)


def NeuralTraining(data_loader, models, optim, loss_fn):
    length = len(data_loader.dataset)
    aver = 0
    num = 0
    for count, (X, y) in enumerate(data_loader):
        num += 1
        pred = models(X)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        aver += loss.item()

        if num == 100:
            loss, current = aver / 100, count * len(X)
            print(f'avg loss: {loss:>7f} [{current}/{length}]')
            aver = 0
            num = 0


def testNeuralNetwork(data_loader, models, loss_fn):
    size = len(data_loader)
    with torch.no_grad():
        avg_value = 0
        for X, y in data_loader:
            pred = models(X)
            avg_value += loss_fn(pred, y)
        print(f'Avg. loss: {avg_value / size:>5f}')


model = NeuralNetwork()

optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
loss_function = torch.nn.L1Loss()

epochs = 100

rows = 2
column = 2

model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# for i in range(epochs + 1):
#     print(f'Epochs {i}\n---------------------------------------------------------')
#     NeuralTraining(data_loader=loader_learn_data, models=model, optim=optimizer, loss_fn=loss_function)
#     testNeuralNetwork(data_loader=loader_test_data, models=model, loss_fn=loss_function)
#     if not (i % 50 == 0 and i != 0):
#         continue
for i in range(4):
    idx = random.randint(0, len(loader_test1_data) - 1)
    un = 0
    with torch.no_grad():
        for (X, y) in loader_test1_data:
            if un == idx:
                x_model = model(X)
                x_pix = x_model[0][0] * 128
                z_pix = x_model[0][1] * 128
                width = x_model[0][2] * 128
                height = x_model[0][3] * 128

                X = loader_test1_data.dataset.__getitem__(idx)
                # print(f'1: {y} \n {X[1]}')\
                print(f'X: {x_pix} Z: {z_pix}')
                print(f'X1: {y[0][0] * 128:>2f} {y[0][1] * 128:>2f}')
                image = transforms.ToPILImage()(X[0])
                buff, ax = plt.subplots(1)
                ax.imshow(image)
                plt.axis('off')
                rec = patches.Rectangle((int(x_pix), int(z_pix)), int(width), int(height), linewidth=1,
                                        edgecolor='r',
                                        facecolor="none")
                ax.add_patch(rec)
                break
            else:
                un += 1

plt.show()

# torch.save(model.state_dict(), 'model_weights.pth')
