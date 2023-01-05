from pathlib import Path
import pathlib
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt


class ImageTransform():
    def __init__(self, resize, mean, std) -> None:
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize,
                    scale=(0.5, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


def display_image_grid(images_filepaths: list[str],
                       predicted_labels=(), cols=5):
    rows = len(images_filepaths)
    figure, ax = plt.subplots(
        nrows=rows, ncols=cols, figsize=(12, 6))

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


def run():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 'cpu'
    )
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

    return
