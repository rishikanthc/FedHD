import pickle
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from collections import defaultdict
from itertools import cycle
import random


class FaceDataset(Dataset):
    def __init__(self, datadir, transform=None):
        folder = "FACE"
        datapath = os.path.join(datadir, folder, "face.pickle")
        self.transform = transform

        with open(datapath, "rb") as f:
            data = pickle.load(f)

        self.data = np.array(data[0])
        self.labels = np.array(data[1])

        # if train:
        #     self.data = data_x[: int(0.75 * len(data_x))]
        #     self.labels = data_y[: int(0.75 * len(data_y))]
        # else:
        #     self.data = data_x[int(0.75 * len(data_x)) :]
        #     self.labels = data_y[int(0.75 * len(data_y)) :]

        # print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            datum = self.transform(datum)

        return torch.from_numpy(datum), torch.tensor(label)


class PAMAP2Dataset(Dataset):
    def __init__(self, datadir, transform=None):
        folder = "pamap2"
        datapath = os.path.join(datadir, folder, "pamap2.pickle")
        self.transform = transform

        with open(datapath, "rb") as f:
            data = pickle.load(f)

        data_x = np.array(data[0])
        data_y = np.array(data[1])

        # if train:
        #     self.data = data_x[: int(0.75 * len(data_x))]
        #     self.labels = data_y[: int(0.75 * len(data_y))]
        # else:
        #     self.data = data_x[int(0.75 * len(data_x)) :]
        #     self.labels = data_y[int(0.75 * len(data_y)) :]

        print(len(data), data_x.shape, np.max(data_y))
        exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            datum = self.transform(datum)

        return torch.from_numpy(datum), torch.tensor(label)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def separate_by_class(dataset):
    class_samples = defaultdict(list)
    for data, label in dataset:
        class_samples[label].append((data, label))
    return class_samples


def distribute_data(class_samples, n_devices):
    device_datasets = {}
    device_data = defaultdict(list)
    device_labels = defaultdict(list)
    devices_cycle = cycle(range(n_devices))

    for label, samples in class_samples.items():
        random.shuffle(samples)
        for device, (sample, label) in zip(devices_cycle, samples):
            device_data[device].append(sample)
            device_labels[device].append(label)

    for device in range(n_devices):
        device_datasets[device] = CustomDataset(
            device_data[device], device_labels[device]
        )

    return device_datasets
