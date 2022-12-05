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


def load_wbcd(p, seedData, q_pct_of_data, nTask, testSplitIdx, seedSplit=10, nSplit=4, nComponentPCA=1, doPCA=False, doShuffle=True):
    np.random.seed(seedData)
    if 'breast_cancer' not in os.listdir('./data'):
        os.mkdir('./data/breast_cancer')

    breast_caner_data = []
    mapping = {"M": 1, "B": 0}
    with open("./data/breast_cancer/wdbc.data", "r") as f:
        for line in f:
            tmp_line = line.strip().split(",")
            breast_caner_data.append(np.expand_dims(np.array([float(feature) for feature in tmp_line[2:]] + [mapping[tmp_line[1]]]), 0))
    breast_caner_data = np.concatenate((breast_caner_data), axis=0)

    print(f"Number of null values in the data: {sum(sum(np.isnan(breast_caner_data)))}")

    # Splitting data into train and test
    X = breast_caner_data[:, :-1]
    y = breast_caner_data[:, -1:]
    kf = KFold(n_splits=nSplit, random_state=seedSplit, shuffle=doShuffle)
    testSplitIdxTemp = 0
    for train_index, test_index in kf.split(X):
        if testSplitIdx==testSplitIdxTemp:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break
        else:
            testSplitIdxTemp+=1

    dataTasks_X = {} # Dictionary containing data of each task
    dataTasks_y = {} # Dictionary containing data of each task
    minNDataTask = float("inf") # Minimum number of samples in tasks
    if nTask==2:
        for idxTask in range(nTask):
            dataTasks_X[idxTask] = X_train[(y_train.flatten()==idxTask)]
            dataTasks_y[idxTask] = y_train[(y_train.flatten()==idxTask)]
            minNDataTask = min(minNDataTask, dataTasks_X[idxTask].shape[0])
            print(f'Number of samples of task {idxTask+1}/{nTask}: {minNDataTask}')
    elif nTask == 1:
        dataTasks_X[0] = X_train
        dataTasks_y[0] = y_train
        minNDataTask = min(minNDataTask, dataTasks_X[0].shape[0])
    else:
        assert 1==0, "nTask!"
    # Number of instances from each task after selecting q_pct_of_data portion of it
    n_m = int(minNDataTask * q_pct_of_data)  # smaller scale version

    ## Mix individual tasks with overall task
    # each worker m gets (1-p)*n=examples from specific tasks and p*n from mixture of all tasks.
    # So p=1 -> homogeneous (zeta = 0); p=0 -> heterogeneous
    train_features_by_machine = []
    train_labels_by_machine = []
    n_individual = int(np.round(n_m * (1. - p)))  # int (1-p)*n_m
    n_all = n_m - n_individual  # int p*n_m
    for idxTask in range(nTask):
        if nTask==1 and q_pct_of_data==1 and p==0:
            task_m_idxs = np.arange(dataTasks_X[idxTask].shape[0])
        else:
            task_m_idxs = np.random.choice(dataTasks_X[idxTask].shape[0],
                                           size=n_individual)  # specific: randomly choose (1-p)*2n_m examples from 2*n_m = 10,842 examples for task m (one (e,o) pair)
        all_tasks_idxs = np.random.choice(X_train.shape[0], size=n_all)  # mixture of tasks: randomly choose p*2n_m examples from all 54,210 examples (all digits)
        data_for_m_X = np.concatenate([dataTasks_X[idxTask][task_m_idxs, :], X_train[all_tasks_idxs, :]], axis=0)  # machine m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        data_for_m_y = np.concatenate([dataTasks_y[idxTask][task_m_idxs, :], y_train[all_tasks_idxs, :]], axis=0)  # machine m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        # train_features_by_machine.append(data_for_m[:, :-1])
        if doPCA:
            data_for_m_X = dimReductionPCA(data_for_m_X, nComponentPCA)
        data_for_m_idx = np.arange(data_for_m_X.shape[0])
        if doShuffle:
            np.random.shuffle(data_for_m_idx)
        train_features_by_machine.append(data_for_m_X[data_for_m_idx, :])
        train_labels_by_machine.append(data_for_m_y[data_for_m_idx, :])
    train_features_by_machine = np.array(train_features_by_machine)  # array of all 8 feauture sets
    train_labels_by_machine = np.array(train_labels_by_machine)  # array of corresponding label sets

    test_features = []
    test_labels = []
    test_labels.append(y_test)
    if doPCA:
        X_test = dimReductionPCA(X_test, nComponentPCA)
    test_features.append(X_test)

    # normalizer = minMaxScaling
    normalizer = standardNormalization
    for idxTask in range(nTask):
        train_features_by_machine[idxTask] = normalizer(train_features_by_machine[idxTask])
    test_features[0] = normalizer(test_features[0])

    return train_features_by_machine, train_labels_by_machine, test_features, test_labels, n_m


class MLPCrossEntropyBC(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out):
        super(MLPCrossEntropyBC, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden1)
        self.layer_hidden1 = nn.Linear(dim_hidden1, dim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        return x
    
    
 
def load_MNIST2(args, seedData, path, dimPCA=50, doPCA=True):
    np.random.seed(seedData)
    if path not in os.listdir('./data'):
        os.mkdir('./data/' + path)
    # if data folder is not there, make one and download/preprocess mnist:
    if 'processed_mnist_features_{:d}.npy'.format(dimPCA) not in os.listdir('./data/' + path):
        # convert image to tensor and normalize (mean 0, std dev 1):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,)), ])
        # download and store transformed dataset as variable mnist:
        mnist = datasets.MNIST('data', download=True, train=True, transform=transform)
        # separate features from labels and reshape features to be 1D array:
        features = np.array([np.array(mnist[i][0]).reshape(-1) for i in range(len(mnist))])
        labels = np.array([mnist[i][1] for i in range(len(mnist))])
        # apply PCA to features to reduce dimensionality to dimPCA
        if doPCA:
            features = PCA(n_components=dimPCA).fit_transform(features)
        # save processed features in "data" folder:
        np.save('data/' + path + '/processed_mnist_features_{:d}.npy'.format(dimPCA), features)
        np.save('data/' + path + '/processed_mnist_labels_{:d}.npy'.format(dimPCA), labels)
    # else (data is already there), load data:
    else:
        features = np.load('data/' + path + '/processed_mnist_features_{:d}.npy'.format(dimPCA))
        labels = np.load('data/' + path + '/processed_mnist_labels_{:d}.npy'.format(dimPCA))

    ## Group the data by digit
    # n_m = smallest number of occurences of any one digit (label) in mnist:
    n_m = int(min([np.sum(labels == i) for i in range(10)]) * args.q_pct_of_data)  # smaller scale version
    # use defaultdict to avoid key error https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work:
    by_number = defaultdict(list)
    # append feature vectors to by_number until there are n_m of each digit
    for i, feat in enumerate(features):
        if len(by_number[labels[i]]) < n_m:
            by_number[labels[i]].append(feat)
    # convert each list of n_m feature vectors (for each digit) in by_number to np array
    for i in range(10):
        by_number[i] = np.array(by_number[i])

    ## Enumerate the even vs. odd tasks
    even_numbers = [0, 2, 4, 6, 8]
    odd_numbers = [1, 3, 5, 7, 9]
    # make list of all 25 pairs of (even, odd):
    even_odd_pairs = list(itertools.product(even_numbers, odd_numbers))

    ## Group data into 25 single even vs single odd tasks
    all_tasks = []
    for (e, o) in even_odd_pairs:
        # eo_feautres: concatenate even feats, odd feats for each e,o pair:
        eo_features = np.concatenate([by_number[e], by_number[o]], axis=0)
        # (0,...,0, 1, ... ,1) labels of length 2*n_m corresponding to eo_features:
        eo_labels = np.concatenate([np.ones(n_m), np.zeros(n_m)])
        # concatenate eo_feautures and eo_labels into array of length 4*n_m:
        eo_both = np.concatenate([eo_labels.reshape(-1, 1), eo_features], axis=1)
        # add eo_both to all_tasks:
        all_tasks.append(eo_both)
    # all_tasks is a list of 25 ndarrays, each array corresponds to an (e,o) pair of digits (aka task) and is 10,842 (examples) x 101 (100=dim (features) plus 1 =dim(label))
    # all_evens: concatenated array of 5*n_m ones and 5*n_m = 27,105 feauture vectors (n_m for each even digit):
    all_evens = np.concatenate([np.ones((5 * n_m, 1)), np.concatenate([by_number[i] for i in even_numbers], axis=0)],
                               axis=1)
    # all_odds: same thing but for odds and with zeros instead of ones:
    all_odds = np.concatenate([np.zeros((5 * n_m, 1)), np.concatenate([by_number[i] for i in odd_numbers], axis=0)],
                              axis=1)
    # combine all_evens and _odds into all_nums (contains all 10*n_m = 54210 training examples):
    all_nums = np.concatenate([all_evens, all_odds], axis=0)

    ## Mix individual tasks with overall task
    # each worker m gets (1-p)* 2*n = (1-p)*10,842 examples from specific tasks and p*10,842 from mixture of all tasks.
    # So p=1 -> homogeneous (zeta = 0); p=0 -> heterogeneous
    features_by_machine = []
    labels_by_machine = []
    n_individual = int(np.round(2 * n_m * (1. - args.p)))  # int (1-p)*2n_m = (1-p)*10,842
    n_all = 2 * n_m - n_individual  # =int p*2n_m  = p*10,842
    for m, task_m in enumerate(all_tasks):  # m is btwn 0 and 24 inclusive
        task_m_idxs = np.random.choice(task_m.shape[0],
                                       size=n_individual)  # specific: randomly choose (1-p)*2n_m examples from 2*n_m = 10,842 examples for task m (one (e,o) pair)
        all_nums_idxs = np.random.choice(all_nums.shape[0],
                                         size=n_all)  # mixture of tasks: randomly choose p*2n_m examples from all 54,210 examples (all digits)
        data_for_m = np.concatenate([task_m[task_m_idxs, :], all_nums[all_nums_idxs, :]],
                                    axis=0)  # machine m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        features_by_machine.append(data_for_m[:, 1:])
        labels_by_machine.append(data_for_m[:, 0])
    features_by_machine = np.array(features_by_machine)  # array of all 25 feauture sets (each set has 10,842 feauture vectors)
    labels_by_machine = np.array(labels_by_machine)  # array of corresponding label sets
    ###Train/Test split for each machine###
    train_features_by_machine = []
    test_features_by_machine = []
    train_labels_by_machine = []
    test_labels_by_machine = []
    for m, task_m in enumerate(all_tasks):
        train_feat, test_feat, train_label, test_label = train_test_split(features_by_machine[m], labels_by_machine[m],
                                                                          test_size=1/args.nSplit, random_state=1)
        train_features_by_machine.append(train_feat)
        test_features_by_machine.append(test_feat)
        train_labels_by_machine.append(train_label)
        test_labels_by_machine.append(test_label)
    train_features_by_machine = np.array(train_features_by_machine)
    test_features_by_machine = np.array(test_features_by_machine)
    train_labels_by_machine = np.array(train_labels_by_machine)
    test_labels_by_machine = np.array(test_labels_by_machine)
    print(train_features_by_machine.shape)
    return train_features_by_machine, train_labels_by_machine, test_features_by_machine, test_labels_by_machine, n_m

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

