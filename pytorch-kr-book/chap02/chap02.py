from functools import cache
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from pathlib import Path


class Model(nn.Module):
    def __init__(self, embedding_size, output_size, layers, p=0.4) -> None:
        super().__init__()
        self.all_embeddings = nn.ModuleList(
            [nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x


def run():
    data_path = Path(str(Path.cwd())+'/chap02/data/car_evaluation.csv')
    dataset = pd.read_csv(data_path)

    plt.figure(figsize=(8, 6))
    dataset['output'].value_counts().plot.pie(
        autopct='%0.05f%%', explode=(0.05, 0.05, 0.05, 0.05))
    plt.show()

    categorical_columns = ['price', 'maint',
                           'doors', 'persons', 'lug_capacity', 'safety']

    for category in categorical_columns:
        dataset[category] = dataset[category].astype('category')

    price = dataset['price'].cat.codes.values
    maint = dataset['maint'].cat.codes.values
    doors = dataset['doors'].cat.codes.values
    persons = dataset['persons'].cat.codes.values
    lug_capacity = dataset['lug_capacity'].cat.codes.values
    safety = dataset['safety'].cat.codes.values

    categorical_data = np.stack(
        [price, maint, doors, persons, lug_capacity, safety], 1)
    print(categorical_data[:10])

    categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
    print(categorical_data[:10])

    outputs = pd.get_dummies(dataset['output'])
    outputs = torch.tensor(outputs.values).flatten()
    print(categorical_data.shape)
    print(outputs.shape)

    categorical_column_sizes = [
        len(dataset[column].cat.categories) for column in categorical_columns]
    categorical_embedding_sizes = [
        (col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
    print(categorical_embedding_sizes)

    total_records = 1728
    test_records = int(total_records * 0.2)

    categorical_train_data = categorical_data[:(total_records-test_records)]
    categorical_test_data = categorical_data[(
        total_records-test_records):total_records]
    train_outputs = outputs[:total_records-test_records]
    test_outputs = outputs[total_records-test_records: total_records]

    print(len(categorical_train_data))
    print(len(train_outputs))
    print(len(categorical_test_data))
    print(len(test_outputs))

    return
