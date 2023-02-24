import math
import numpy as np
import os
import itertools
from collections import defaultdict
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import torchvision.datasets as datasets
import pickle
import torch.nn.functional as F


def load_cifar10_data(args, seedData, doShuffle=True):
    np.random.seed(seedData)

    trainTrasnform = [transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))]
    cifar_trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transforms.Compose(trainTrasnform))
    testTrasnform = [transforms.ToTensor(), transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2467, 0.2429, 0.2616))]
    cifar_testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transforms.Compose(testTrasnform))

    if 'cifar10_transformed.pk' not in os.listdir('./data/cifar10'):
        trainTrasnform = [transforms.ToTensor(),
                          transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))]
        cifar_trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True,
                                          transform=transforms.Compose(trainTrasnform))
        testTrasnform = [transforms.ToTensor(),
                         transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2467, 0.2429, 0.2616))]
        cifar_testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True,
                                         transform=transforms.Compose(testTrasnform))

        X_train = torch.stack([img_t for idx, (img_t, _) in enumerate(cifar_trainset)], dim=0).numpy()
        y_train = torch.stack([torch.tensor(label_t).view(1) for idx, (_, label_t) in enumerate(cifar_trainset)],
                              dim=0).numpy()
        X_test = torch.stack([img_t for img_t, _ in cifar_testset], dim=0).numpy()
        y_test = torch.stack([torch.tensor(label_t).view(1) for _, label_t in cifar_testset], dim=0).numpy()

        with open('./data/cifar10/cifar10_transformed.pk', 'wb') as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)
    else:
        with open('./data/cifar10/cifar10_transformed.pk', 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)

    dataTasks_X = {} # Dictionary containing data of each task
    dataTasks_y = {} # Dictionary containing data of each task
    nClass = np.size(np.unique(y_train))
    assert args.M%nClass==0, f"Number of tasks should be multiply of {nClass}"
    nMachinePerClass = int(args.M/nClass)
    for idxClass in range(nClass):
        idxDataClass = np.where(y_train[:, 0]==idxClass)[0]
        np.random.shuffle(idxDataClass)
        nDataClass = np.shape(idxDataClass)[0]
        # To be able to split indecies into subarrays with same size, we remove some samples
        idxRemovedSamples = nDataClass%nMachinePerClass
        idxDataClassSplit = np.split(idxDataClass[:nDataClass-idxRemovedSamples], indices_or_sections=nMachinePerClass, axis=0)
        for idxSplit in range(nMachinePerClass):
            idxMachine = idxClass*nMachinePerClass+idxSplit
            idxDataMachine = idxDataClassSplit[idxSplit]
            dataTasks_X[idxMachine] = X_train[idxDataMachine]
            dataTasks_y[idxMachine] = y_train[idxDataMachine]

    n_m = int(np.shape(dataTasks_X[0])[0] * args.q_pct_of_data)  # smaller scale version

    ## Mix individual tasks with overall task
    # each worker m gets (1-p)*n=examples from specific tasks and p*n from mixture of all tasks.
    # So p=1 -> homogeneous (zeta = 0); p=0 -> heterogeneous
    train_features_by_machine = []
    train_labels_by_machine = []
    n_individual = int(np.round(n_m * (1. - args.p)))  # int (1-p)*n_m
    n_all = n_m - n_individual  # int p*n_m
    for idxTask in range(args.M):
        task_m_idxs = np.random.choice(dataTasks_X[idxTask].shape[0],
                                       size=n_individual)  # specific: randomly choose (1-p)*2n_m examples from 2*n_m = 10,842 examples for task m (one (e,o) pair)
        all_tasks_idxs = np.random.choice(X_train.shape[0], size=n_all)  # mixture of tasks: randomly choose p*2n_m examples from all 54,210 examples (all digits)
        data_for_m_X = np.concatenate([dataTasks_X[idxTask][task_m_idxs, :], X_train[all_tasks_idxs, :]], axis=0)  # machine m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        data_for_m_y = np.concatenate([dataTasks_y[idxTask][task_m_idxs, :], y_train[all_tasks_idxs, :]], axis=0)  # machine m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        data_for_m_idx = np.arange(data_for_m_X.shape[0])
        if doShuffle:
            np.random.shuffle(data_for_m_idx)
        train_features_by_machine.append(data_for_m_X[data_for_m_idx, :])
        train_labels_by_machine.append(data_for_m_y[data_for_m_idx, :])
    train_features_by_machine = np.array(train_features_by_machine)  # array of all feauture sets
    train_labels_by_machine = np.array(train_labels_by_machine)  # array of corresponding label sets

    test_features = []
    test_labels = []
    test_features.append(X_test)
    test_labels.append(y_test)

    return train_features_by_machine, train_labels_by_machine, test_features, test_labels, n_m


class Net_CIFAR10_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def minMaxScaling(dataInput):
    dataOutput = np.zeros_like(dataInput)
    nFeature = dataInput.shape[1]
    for idxFeature in range(nFeature):
        minVal = np.min(dataInput[:, idxFeature])
        maxVal = np.max(dataInput[:, idxFeature])
        if minVal == maxVal:
            dataInput[:, idxFeature] = 1
            continue
        dataOutput[:, idxFeature] = (dataInput[:, idxFeature] - minVal) / (maxVal - minVal)
    return dataOutput


def standardNormalization(dataInput):
    scaler = StandardScaler()
    scaler.fit(dataInput)
    return scaler.transform(dataInput)


def dimReductionPCA(data, nComponent):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=nComponent)
    pca.fit(data)
    data = pca.transform(data)
    return data
