import torch
import torchtext
import torchdata
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time


def run():
    start = time.time()
    train_iter, test_iter = torchtext.datasets.IMDB(split=('train', 'test'))
    

    return
