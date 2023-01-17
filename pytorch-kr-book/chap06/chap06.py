from pathlib import Path
import pathlib
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from torchsummary import summary


class ImageTransform():
    def __init__(self, size, mean, std) -> None:
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    size,
                    scale=(0.5, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


def display_image_grid(images_filepaths: list[str],
                       predicted_labels=(), cols=5):
    rows = len(images_filepaths)
    print(f'{rows} : rows')
    figure, ax = plt.subplots(
        nrows=2, ncols=cols, figsize=(12, 6))

    for i, image_filepath in enumerate(images_filepaths):
        print(image_filepath)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = image_filepath.split('\\')[-2]
        predicted_label = (predicted_labels[i]
                           if predicted_labels
                           else true_label)
        color = ('green'
                 if true_label == predicted_label
                 else 'red')
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

    return


class DogvsCatDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list: list[str] = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        label = img_path.split('\\')[-1].split('.')[0]
        label = 1 if label == 'dog' else 0

        return img_transformed, label


class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16,
                              kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32*53*53, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out: torch.Tensor = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.output(out)
        return out


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)


def train_model(model, dataloader_dict,
                criterion, optimizer, num_epoch, device):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model


def run_LeNet():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 'cpu'
    )
    device = 'cpu'
    cat_directory = (str(Path.cwd()) +
                     '/chap06/data/dogs-vs-cats/Cat/')
    dog_directory = (str(Path.cwd()) +
                     '/chap06/data/dogs-vs-cats/Dog/')
    cat_images_filepaths = sorted(
        [str(f) for f in Path(cat_directory).iterdir()
         if f.is_file()]
    )
    dog_images_filepaths = sorted(
        [str(f) for f in Path(dog_directory).iterdir()
         if f.is_file()]
    )

    images_filepaths = [*cat_images_filepaths,
                        *dog_images_filepaths]

    correct_images_filepaths = [
        i for i in images_filepaths
        if cv2.imread(i) is not None
    ]

    random.seed(42)
    random.shuffle(correct_images_filepaths)
    train_images_filepaths = (
        correct_images_filepaths[:400]
    )
    val_images_filepaths = (
        correct_images_filepaths[400:-10]
    )
    test_images_filepaths = (
        correct_images_filepaths[-10:]
    )

    print(len(train_images_filepaths),
          len(val_images_filepaths),
          len(test_images_filepaths))

    display_image_grid(test_images_filepaths)

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32

    train_dataset = DogvsCatDataset(
        train_images_filepaths,
        transform=ImageTransform(size=size, mean=mean, std=std),
        phase='train')

    val_dataset = DogvsCatDataset(
        val_images_filepaths,
        transform=ImageTransform(size=size, mean=mean, std=std),
        phase='val')

    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {'train': train_dataloader,
                       'val': val_dataloader}
    batch_iterator = iter(train_dataloader)
    inputs, label = next(batch_iterator)
    print(inputs.size())
    print(label)

    model = LeNet()
    model = model.to(device=device)
    print(model)
    summary(model, input_size=(3, 224, 224), device=device)
    print(
        f'The model has {count_parameters(model=model):,} trainable parameters')

    optimizer = optim.SGD(model.parameters(),
                          lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    num_epoch = 10
    model: nn.Module = train_model(model=model,
                                   dataloader_dict=dataloader_dict,
                                   criterion=criterion, num_epoch=num_epoch,
                                   optimizer=optimizer, device=device)

    id_list = []
    pred_list = []
    _id = 0
    with torch.no_grad():
        for test_path in tqdm(test_images_filepaths):
            img = Image.open(test_path)
            _id = test_path.split('\\')[-1].split('.')[1]
            transforms = ImageTransform(size, mean, std)
            img = transforms(img, phase='val')
            img = img.unsqueeze(0)
            img = img.to(device)

            model.eval()
            outputs = model(img)
            preds = F.softmax(outputs, dim=1)[:, 1].tolist()
            id_list.append(_id)
            pred_list.append(preds[0])

    res = pd.DataFrame({
        'id': id_list,
        'label': pred_list
    })

    res.sort_values(by='id', inplace=True)
    res.reset_index(drop=True, inplace=True)

    res.to_csv(
        str(Path.cwd()) +
        '/chap06/data/LeNet', index=False
    )

    return


def run():
    run_LeNet()
    return
