import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

import os
import time
import copy
import glob
import cv2
import shutil

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from matplotlib.figure import Figure


class FashionDNN(nn.Module):
    def __init__(self) -> None:
        super(FashionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, input_data: torch.Tensor):
        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class FashionCNN(nn.Module):
    def __init__(self) -> None:
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def run_FashionDNN(device: torch.device, train_loader: DataLoader, test_loader: DataLoader):
    learning_rate = 0.001
    model = FashionDNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)

    num_epochs = 5
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)
            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)

            outputs = model(train)
            loss: torch.Tensor = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            if not (count % 50):
                total = 0
                correct = 0
                for images, labels in test_loader:
                    images: torch.Tensor = images.to(device)
                    labels: torch.Tensor = labels.to(device)
                    labels_list.append(labels)
                    test = Variable(images.view(100, 1, 28, 28))
                    outputs = model(test)
                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print(
                    f'Iteration: {count}, Loss: {loss.data}, Accuracy: {accuracy}%')


def run_FashionCNN(device: torch.device, train_loader: DataLoader, test_loader: DataLoader):
    learning_rate = 0.001
    model = FashionCNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)

    num_epochs = 5
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)
            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)

            outputs = model(train)
            loss: torch.Tensor = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            if not (count % 50):
                total = 0
                correct = 0
                for images, labels in test_loader:
                    images: torch.Tensor = images.to(device)
                    labels: torch.Tensor = labels.to(device)
                    labels_list.append(labels)
                    test = Variable(images.view(100, 1, 28, 28))
                    outputs = model(test)
                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()
                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print(
                    f'Iteration: {count}, Loss: {loss.data}, Accuracy: {accuracy}%')


def run_basic_cnn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    data_path = str(Path(str(Path.cwd())+'/chap05/data'))
    train_dataset = torchvision.datasets.FashionMNIST(
        data_path, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_dataset = torchvision.datasets.FashionMNIST(
        data_path, download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=100)
    test_loader = DataLoader(test_dataset, batch_size=100)

    labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
                  4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

    fig: Figure = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows + 1):
        img_xy = np.random.randint(len(train_dataset))
        img = train_dataset[img_xy][0][0, :, :]
        fig.add_subplot(rows, columns, i)
        plt.title(labels_map[train_dataset[img_xy][1]])
        plt.axis('off')
        plt.imshow(img, cmap='gray')

    plt.show()

#    run_FashionDNN(device=device, train_loader=train_loader,
#                   test_loader=test_loader)
    run_FashionCNN(device=device, train_loader=train_loader,
                   test_loader=test_loader)

    return


def train_model(model: nn.Module, dataloaders: DataLoader,  optimizer: torch.optim.Optimizer,
                device: torch.device, criterion=torch.nn.CrossEntropyLoss, num_epochs=13, is_train=True):
    since = time.time()
    acc_history = []
    loss_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print(f'Loss:{epoch_loss} Acc:{epoch_acc}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)
        data_path = str(Path(str(Path.cwd())+'/chap05/data/catanddog'))
        torch.save(model.state_dict(), data_path+'/{0:0=2d}.pth'.format(epoch))
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60}m {time_elapsed%60}s')
    print(f'Best Acc: {best_acc}')

    return acc_history, loss_history


def set_parameter_requires_grad(model: nn.Module,
                                feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def run_transfer_learning():
    data_path = str(Path(str(Path.cwd())+'/chap05/data/catanddog/train'))
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        data_path,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4,
                              shuffle=True)
    print(len(train_dataset))

    samples, labels = iter(train_loader).next()
    classes = {0: 'cat', 1: 'dog'}
    fig = plt.figure(figsize=(16, 24))
    for i in range(24):
        a = fig.add_subplot(4, 6, i+1)
        a.set_title(classes[labels[i].item()])
        a.axis('off')
        a.imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(resnet18)
    resnet18.fc = nn.Linear(512, 2)

    for name, param in resnet18.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(512, 2)
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters())
    cost = torch.nn.CrossEntropyLoss()
    print(model)

    params_to_update = []
    for name, param in resnet18.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    optimizer = optim.Adam(params_to_update)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    train_acc_hist, train_loss_hist = train_model(model=resnet18, dataloaders=train_loader,
                                                  criterion=criterion, optimizer=optimizer, device=device)

    return


def run():
    # run_basic_cnn()
    run_transfer_learning()

    return
