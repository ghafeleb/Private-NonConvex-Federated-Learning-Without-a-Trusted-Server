#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import os
import itertools
from collections import defaultdict
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from Nets import MLP
import argparse
import copy
import time
import scipy.io as sio
import matlab.engine

np_seed = 1
np.set_printoptions(precision=3, linewidth=240, suppress=True)
np.random.seed(np_seed)

q = 1/7 #fraction of mnist data we wish to use; q = 1 -> 8673 train examples per machine; q = 1/10 -> 867 train examples per machine
###Function to download and pre-process (normalize, PCA) mnist and store in "data" folder:
    ##Returns 4 arrays: train/test_features_by_machine = , train/test_labels_by_machine 
def load_MNIST2(p, dim, path, model):
    if path not in os.listdir('./data'):
        os.mkdir('./data/'+path)
    #if data folder is not there, make one and download/preprocess mnist:
    if 'processed_mnist_features_{:d}.npy'.format(dim) not in os.listdir('./data/'+path):
        #convert image to tensor and normalize (mean 0, std dev 1):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.,), (1.,)),])
        #download and store transformed dataset as variable mnist:
        mnist = datasets.MNIST('data', download=True, train=True, transform=transform)
        #separate features from labels and reshape features to be 1D array:
        features = np.array([np.array(mnist[i][0]).reshape(-1) for i in range(len(mnist))])
        labels = np.array([mnist[i][1] for i in range(len(mnist))])
        #apply PCA to features to reduce dimensionality to dim
        features = PCA(n_components=dim).fit_transform(features)
        #save processed features in "data" folder:
        np.save('data/' + path + '/processed_mnist_features_{:d}.npy'.format(dim), features)
        np.save('data/' + path + '/processed_mnist_labels_{:d}.npy'.format(dim), labels)
    #else (data is already there), load data:
    else:
        features = np.load('data/' + path + '/processed_mnist_features_{:d}.npy'.format(dim))
        labels = np.load('data/' + path + '/processed_mnist_labels_{:d}.npy'.format(dim))
    
    ## Group the data by digit
    #n_m = smallest number of occurences of any one digit (label) in mnist:
    n_m = int(min([np.sum(labels == i) for i in range(10)])*q) #smaller scale version 
    #use defaultdict to avoid key error https://stackoverflow.com/questions/5900578/how-does-collections-defaultdict-work:
    by_number = defaultdict(list)
    #append feature vectors to by_number until there are n_m of each digit
    for i, feat in enumerate(features):
        if len(by_number[labels[i]]) < n_m:
            by_number[labels[i]].append(feat)
    #convert each list of n_m feature vectors (for each digit) in by_number to np array 
    for i in range(10):
        by_number[i] = np.array(by_number[i])

    ## Enumerate the even vs. odd tasks
    even_numbers = [0,2,4,6,8]
    odd_numbers = [1,3,5,7,9]
    #make list of all 25 pairs of (even, odd):
    even_odd_pairs = list(itertools.product(even_numbers, odd_numbers))

    ## Group data into 25 single even vs single odd tasks
    all_tasks = []
    for (e,o) in even_odd_pairs:
        #eo_feautres: concatenate even feats, odd feats for each e,o pair:
        eo_features = np.concatenate([by_number[e], by_number[o]], axis=0)
        #(0,...,0, 1, ... ,1) labels of length 2*n_m corresponding to eo_features:
        eo_labels = np.concatenate([np.ones(n_m), np.zeros(n_m)])
        #concatenate eo_feautures and eo_labels into array of length 4*n_m:
        eo_both = np.concatenate([eo_labels.reshape(-1,1), eo_features], axis=1)
        #add eo_both to all_tasks:
        all_tasks.append(eo_both)
    #all_tasks is a list of 25 ndarrays, each array corresponds to an (e,o) pair of digits (aka task) and is 10,842 (examples) x 101 (100=dim (features) plus 1 =dim(label))
    #all_evens: concatenated array of 5*n_m ones and 5*n_m = 27,105 feauture vectors (n_m for each even digit):
    all_evens = np.concatenate([np.ones((5*n_m,1)), np.concatenate([by_number[i] for i in even_numbers], axis=0)], axis=1)
    #all_odds: same thing but for odds and with zeros instead of ones:
    all_odds = np.concatenate([np.zeros((5*n_m,1)), np.concatenate([by_number[i] for i in odd_numbers], axis=0)], axis=1)
    #combine all_evens and _odds into all_nums (contains all 10*n_m = 54210 training examples):
    all_nums = np.concatenate([all_evens, all_odds], axis=0)

    ## Mix individual tasks with overall task
    #each worker m gets (1-p)* 2*n = (1-p)*10,842 examples from specific tasks and p*10,842 from mixture of all tasks.
    #So p=1 -> homogeneous (zeta = 0); p=0 -> heterogeneous
    features_by_machine = []
    labels_by_machine = []
    n_individual = int(np.round(2*n_m * (1. - p))) #int (1-p)*2n_m = (1-p)*10,842
    n_all = 2*n_m - n_individual #=int p*2n_m  = p*10,842
    for m, task_m in enumerate(all_tasks): #m is btwn 0 and 24 inclusive
        task_m_idxs = np.random.choice(task_m.shape[0], size = n_individual) #specific: randomly choose (1-p)*2n_m examples from 2*n_m = 10,842 examples for task m (one (e,o) pair)
        all_nums_idxs = np.random.choice(all_nums.shape[0], size = n_all) #mixture of tasks: randomly choose p*2n_m examples from all 54,210 examples (all digits)
        data_for_m = np.concatenate([task_m[task_m_idxs, :], all_nums[all_nums_idxs, :]], axis=0) #machine m gets 10,842 total examples: fraction p are mixed, 1-p are specific to task m (one eo pair)
        features_by_machine.append(data_for_m[:,1:])
        labels_by_machine.append(data_for_m[:,0])
    features_by_machine = np.array(features_by_machine) #array of all 25 feauture sets (each set has 10,842 feauture vectors)
    labels_by_machine = np.array(labels_by_machine) #array of corresponding label sets
    ###Train/Test split for each machine###
    train_features_by_machine = []
    test_features_by_machine = []
    train_labels_by_machine = []
    test_labels_by_machine = []
    for m, task_m in enumerate(all_tasks):
        train_feat, test_feat, train_label, test_label = train_test_split(features_by_machine[m], labels_by_machine[m], test_size=0.20, random_state=1)
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


class LocalUpdate(object):
    def __init__(self, args, train_features_local, train_labels_local):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = (train_features_local, train_labels_local)

    def train(self, stepsize, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=stepsize)
        images, labels = self.ldr_train[0].to(self.args.device), self.ldr_train[1].to(self.args.device)
        net.zero_grad()
        log_probs = net(images)
        loss = self.loss_func(log_probs, labels.long())
        loss.backward()
        optimizer.step()
        return net.state_dict(), net.named_parameters(), loss.item()


    def train_local_SGD(self, L, stepsize, net, private=-1, noise_dict={}, noise_idx={}):
        net.train()
        # train and update
        losses = []
        optimizer = torch.optim.SGD(net.parameters(), lr=stepsize)
        data_point_idx = 0
        for image, label in zip(self.ldr_train[0], self.ldr_train[1]):
            image, label = image.to(self.args.device).unsqueeze(0), label.to(self.args.device).unsqueeze(0)
            net.zero_grad()
            log_probs = net(image)
            loss = self.loss_func(log_probs, label.long())
            loss.backward()
            for layer, param in net.named_parameters():
                if args.clipping != -1:
                    c = min(1, L / torch.norm(param.grad))  # clip
                    param.grad *= c
                if private != -1:
                    added_noise = (noise_dict[layer][noise_idx[layer]]).reshape(param.grad.shape).to(args.device).float()
                    noise_idx[layer] += 1
                    param.grad += added_noise
            optimizer.step()
            losses.append(loss.item())
            data_point_idx += 1
        return net.state_dict(), net.named_parameters(), sum(losses)/len(losses)


def gauss_AC(d, eps, delta, n, R, L): #moments account form of noise
    return np.random.multivariate_normal(mean = np.zeros(d), cov = ((8*(L**2)*R*np.log(2/delta))/(n**2 * eps**2))*np.eye(d))


def mlp_local_sgd(args, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel):
    np.random.seed(np_seed)
    counter_increase = 0 # Counts the number of times our loss is larger that loss in iteration 1.
    losses = []
    torch.manual_seed(seedModel)
    net_glob = MLP(dim_in=train_features.shape[-1], dim_hidden=args.dH, dim_out=args.num_classes).to(args.device)
    net_glob.train()
    layers_set = set()
    for layer, param in net_glob.named_parameters():
        print(f"layer: {layer}, size of parameters: {param.size()}")
        # iterates[layer] = [param]
        layers_set.add((layer, param.size()))

    for r in range(R):
        #randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        w_end = {}
        for layer, param in net_glob.named_parameters():
            # print(layer, param.size())
            w_end[layer] = torch.zeros_like(param)
        loss_m = []
        for m in S:
            net_local = copy.deepcopy(net_glob)
            idxs = np.random.randint(0, train_features[m].shape[0], K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            local = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
            eps = 0
            state_dict, named_parameters_local, loss = local.train_local_SGD(L, stepsize, net=net_local.to(args.device))
            loss_m.append(loss)
            for layer, weights in state_dict.items():
                w_end[layer] += (weights)/Mavail   # NOISE IS NOT FOR WEIGHT >> NOISE is for gradient of the local machine during training
        # take average of gradients
        losses.append(sum(loss_m)/Mavail)
        print(f"Loss at round {r}: {losses[-1]}")
        if losses[-1]>losses[0]:
            counter_increase += 1
        if counter_increase>=50 or losses[-1]>losses[0]*10:
            return losses, 'diverged', net_glob
        net_glob.load_state_dict(w_end)

    print('')
    return losses, 'converged', net_glob


def mlp_noisy_local_sgd(args, eps, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel):
    np.random.seed(np_seed)
    counter_increase = 0 # Counts the number of times our loss is larger that loss in iteration 1.
    losses = []
    torch.manual_seed(seedModel)
    net_glob = MLP(dim_in=train_features.shape[-1], dim_hidden=args.dH, dim_out=args.num_classes).to(args.device)
    net_glob.train()
    layers_set = set()
    for layer, param in net_glob.named_parameters():
        print(f"layer: {layer}, size of parameters: {param.size()}")
        # iterates[layer] = [param]
        layers_set.add((layer, param.size()))
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    for r in range(R):
        #randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        w_end = {}
        for layer, param in net_glob.named_parameters():
            w_end[layer] = torch.zeros_like(param)
        loss_m = []
        t = time.time()
        for m in S:
            net_local = copy.deepcopy(net_glob)
            idxs = np.random.randint(0, train_features[m].shape[0], K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            local = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
            noise_dict = {}
            noise_idx = {}
            for layer, param_size in layers_set:
                x_len = np.prod(param_size)
                seedNoiseTemp = np.random.rand(1)
                noise_dict[layer] = torch.tensor(eng.noise_generator_localSGD2(float(x_len), float(eps), float(delta), float(n), float(R), float(L), float(K), float(seedNoiseTemp), nargout=1))
                noise_idx[layer] = 0
            state_dict, named_parameters_local, loss = local.train_local_SGD(L, stepsize, net=net_local.to(args.device), private=1, noise_dict=noise_dict, noise_idx=noise_idx)
            loss_m.append(loss)
            for layer, weights in state_dict.items():
                w_end[layer] += (weights)/Mavail   # NOISE IS NOT FOR WEIGHT >> NOISE is for gradient of the local machine during training
        print(f"Time spent on round {r}: {(time.time() - t)/60} minutes")
        # take average of gradients
        losses.append(sum(loss_m)/Mavail)
        print(f"Loss at round {r}: {losses[-1]}")
        if losses[-1]>losses[0]:
            counter_increase += 1
        if counter_increase>=50 or losses[-1]>losses[0]*10:
            return losses, 'diverged', net_glob

        net_glob.load_state_dict(w_end)
    # Stop engine
    eng.quit()
    print('')
    return losses, 'converged', net_glob


def load_noise(idxNoise, layers_set, noise_file_idx, delta, L, n, R, Mavail, eps, seedNoise):
    noise_idx = {}
    noise_dict = {}
    noise_add = os.path.join("data","noise","w1_3200_b1_64_w2_128_b2_2".format(n, n, n, n),"delta{:.3g}_n{:d}_R{:d}_M{:d}_sN{:d}".format(delta, n, R, Mavail, seedNoise),"eps_{:.2g}_L_{:d}".format(eps, L))
    for layer, param_size in layers_set:
        layerType, weightType = layer.split(".")
        if layerType == "layer_input" and weightType == "weight":
            layerName = "w1"
        elif layerType == "layer_input" and weightType == "bias":
            layerName = "b1"
        elif layerType == "layer_hidden" and weightType == "weight":
            layerName = "w2"
        elif layerType == "layer_hidden" and weightType == "bias":
            layerName = "b2"
        noise_file_name = layerName + "_" + str(np.prod(param_size)) + "_bucket" + str(noise_file_idx) + "_noise" + str(idxNoise) + ".mat"
        noise_dict[layer] = torch.tensor(sio.loadmat(os.path.join(noise_add, noise_file_name))["noise"])
        noise_idx[layer] = 0
    return noise_dict, noise_idx


def mlp_noisy_minibatch_sgd(args, eps, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel, seedNoise):
    counter_increase = 0 # Counts the number of times our loss is larger that loss in iteration 1.
    np.random.seed(np_seed)
    torch.manual_seed(seedModel)
    net_glob = MLP(dim_in=train_features.shape[-1], dim_hidden=args.dH, dim_out=args.num_classes).to(args.device)
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=stepsize)
    losses = []
    # Load noise
    noise_file_idx = 1
    if R%Mavail==0:
        noise_bucket_size = R
    else:
        noise_bucket_size = Mavail
    assert noise_bucket_size%Mavail == 0
    layers_set = set()
    for layer, param in net_glob.named_parameters():
        print(f"layer: {layer}, size of parameters: {param.size()}")
        # iterates[layer] = [param]
        layers_set.add((layer, param.size()))
    noise_dict, noise_idx = load_noise(2, layers_set, noise_file_idx, delta, L, n, R, Mavail, eps, seedNoise)

    for r in range(R):
        #randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        g = {}
        loss_r_temp = 0
        for m in S:
            g[m] = {}
            idxs = np.random.randint(0, train_features[m].shape[0], K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            loss_total = 0
            for idx in idxs:
                local = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                net_local = copy.deepcopy(net_glob)
                _, named_parameters_local, loss = local.train(stepsize, net=net_local.to(args.device))
                for layer, param in named_parameters_local:
                    b = copy.deepcopy(param.grad) / (
                                K * 1.)  # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                    if args.clipping != -1:
                        c = min(1, L / torch.norm(b))  # clip
                        if c>1:
                            print(c)
                        b *= c
                    if layer not in g[m]:
                        g[m][layer] = b
                    else:
                        g[m][layer] += b
                loss_total += loss
            loss_total /= (K*1.)
            # print(f'loss_total of machine {m}: {loss_total}')
            loss_r_temp += loss_total
        # take average of gradients
        losses.append(loss_r_temp/Mavail)
        print(f"Loss at round {r}: {losses[-1]}")
        if losses[-1]>losses[0]:
            counter_increase += 1
        if counter_increase>=50 or losses[-1]>losses[0]*10:
            return losses, 'diverged', net_glob

        g_avg = {}
        for layer, _ in layers_set:
            if noise_idx[layer] >= noise_bucket_size:
                noise_file_idx += 1
                noise_dict, noise_idx = load_noise(2, layers_set, noise_file_idx, delta, L, n, R, Mavail, eps, seedNoise)
        for layer, _ in layers_set:
            t_now = time.time()
            time_start = time.time()
            added_noise = noise_dict[layer][noise_idx[layer]].reshape(g[S[0]][layer].shape).to(args.device).float()
            noise_idx[layer] += 1
            g_avg[layer] = g[S[0]][layer] + added_noise
            tot_time = time.time() - time_start
            for idxClient in S[1:]:
                time_start = time.time()
                added_noise = noise_dict[layer][noise_idx[layer]].reshape(g[idxClient][layer].shape).to(args.device).float()
                noise_idx[layer] += 1
                g_avg[layer] += g[idxClient][layer] + added_noise
                tot_time = time.time() - time_start
            # Take average of aggregated noisy gradients
            g_avg[layer] /= (Mavail * 1.)
            # print(f"Computation time for layer {layer} with {x_len} parameters: {(time.time()-t_now)/60} minute(s)")
        net_glob.zero_grad()
        for layer, param in net_glob.named_parameters():
            param.grad = g_avg[layer]
        optimizer.step()

    print('')
    return losses, 'converged', net_glob


def mlp_minibatch_sgd(args, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel, avg_window=8):
    np.random.seed(np_seed)
    counter_increase = 0 # Counts the number of times our loss is larger that loss in iteration 1.
    torch.manual_seed(seedModel)
    net_glob = MLP(dim_in=train_features.shape[-1], dim_hidden=args.dH, dim_out=args.num_classes).to(args.device)
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=stepsize)
    losses = []

    layers_set = set()
    for layer, param in net_glob.named_parameters():
        print(f"layer: {layer}, size of parameters: {param.size()}")
        layers_set.add((layer, param.size()))

    for r in range(R):
        #randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        g = {}
        loss_r_temp = 0
        for m in S:
            g[m] = {}
            idxs = np.random.randint(0, train_features[m].shape[0], K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            loss_total = 0
            for idx in idxs:
                local = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                net_local = copy.deepcopy(net_glob)
                _, named_parameters_local, loss = local.train(stepsize, net=net_local.to(args.device))
                loss_total += loss
                for layer, param in named_parameters_local:
                    b = copy.deepcopy(param.grad) / (
                                K * 1.)  # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                    if args.clipping != -1:
                        c = min(1, L / torch.norm(b))  # clip
                        b *= c
                    if layer not in g[m]:
                        g[m][layer] = b
                    else:
                        g[m][layer] += b
            loss_total /= (K*1.)
            loss_r_temp += loss_total
        # take average of gradients
        losses.append(loss_r_temp/Mavail)
        print(f"Loss at round {r}: {losses[-1]}")
        if losses[-1]>losses[0]:
            counter_increase += 1
        if counter_increase>=50 or losses[-1]>losses[0]*10:
            return losses, 'diverged', net_glob

        # Taking average of gradients
        g_avg = {}
        for layer, _ in layers_set:
            g_avg[layer] = g[S[0]][layer]
            for idxClient in S[1:]:
                g_avg[layer] += g[idxClient][layer]
            # Take average of aggregated noisy gradients
            g_avg[layer] /= (Mavail * 1.)

        net_glob.zero_grad()
        for layer, param in net_glob.named_parameters():
            param.grad = g_avg[layer]
        optimizer.step()

    print('')
    return losses, 'converged', net_glob


def mlp_noisy_SPIDER(args, eps, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel, seedNoise, avg_window=8):
    np.random.seed(np_seed)
    losses = [] # keeps the loss value at each round
    counter_increase = 0 # Counts the number of times our loss is larger that loss in iteration 1.
    noise1_file_idx = 1 # Index of the bucket of file of noise 1
    noise2_file_idx = 1 # Index of the bucket of file of noise 2
    # noise_bucket_size = R
    if R%Mavail==0:
        noise_bucket_size = R
    else:
        noise_bucket_size = Mavail
    assert noise_bucket_size%Mavail == 0
    for r in range(0, R):
        #randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        if r==0:
            torch.manual_seed(seedModel) # Seed of model
            net_glob_2 = MLP(dim_in=train_features.shape[-1], dim_hidden=args.dH, dim_out=args.num_classes).to(
                args.device) # Initialize global model
            net_glob_2.train()

            layers_set = set() # Set of layers that model has
            for layer, param in net_glob_2.named_parameters():
                print(f"layer: {layer}, size of parameters: {param.size()}")
                # iterates[layer] = [param]
                layers_set.add((layer, param.size()))
            noise1_dict, noise1_idx = load_noise(1, layers_set, noise1_file_idx, delta, L, n, R, Mavail, eps, seedNoise) # Load first bucket of file of noise 1
            noise2_dict, noise2_idx = load_noise(2, layers_set, noise2_file_idx, delta, L, n, R, Mavail, eps, seedNoise) # Load first bucket of file of noise 2

        elif r>=1:
            net_glob_0 = copy.deepcopy(net_glob_2)
            net_glob_0.train()

            optimizer_2 = torch.optim.SGD(net_glob_2.parameters(), lr=stepsize)
            net_glob_2.zero_grad()
            # for i in list(net_glob.parameters()):
            #     i.grad = torch.Variable(torch.from_numpy(GRADIENT_ARRAY))
            for layer, param in net_glob_2.named_parameters():
                # print(layer, param.size())
                param.grad = g_avg_2[layer]
            optimizer_2.step()

            net_glob_1 = copy.deepcopy(net_glob_2)
            net_glob_1.train()
            optimizer_1 = torch.optim.SGD(net_glob_1.parameters(), lr=stepsize)

            # Line 16 of Algorithm: Compute gradient w.r.t. w^0
            g_0 = {} # g_0 = {'layer1': [...], 'layer2': [...]}
            g_1 = {} # g_1 = {'layer1': [...], 'layer2': [...]}
            for m in S:
                g_0[m] = {}
                g_1[m] = {}
                idxs = np.random.randint(0, train_features[m].shape[0], K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
                for idx in idxs:
                    local0 = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                    net_local0 = copy.deepcopy(net_glob_0)
                    _, named_parameters_local0, _ = local0.train(stepsize, net=net_local0.to(args.device))
                    for layer, param in named_parameters_local0:
                        b = copy.deepcopy(param.grad)/(K*1.) # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                        if args.clipping!=-1:
                            c = min(1, L / torch.norm(b))  # clip
                            b *= c
                        if layer not in g_0[m]:
                            g_0[m][layer] = b
                        else:
                            g_0[m][layer] += b
                    local1 = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                    net_local1 = copy.deepcopy(net_glob_1)
                    _, named_parameters_local1, _ = local1.train(stepsize, net=net_local1.to(args.device))
                    for layer, param in named_parameters_local1:
                        b = copy.deepcopy(param.grad)/(K*1.) # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                        if args.clipping!=-1:
                            c = min(1, L / torch.norm(b))  # clip
                            b *= c
                        if layer not in g_1[m]:
                            g_1[m][layer] = b
                        else:
                            g_1[m][layer] += b

            # Line 16 of Algorithm: Compute v^1
            g_avg_1 = {}
            # d = number of parameters in the layer
            for layer in list(g_1[S[0]].keys()):
                idxNoise = 1
                if noise1_idx[layer] >= noise_bucket_size:
                    noise1_file_idx += 1
                    noise1_dict, noise1_idx = load_noise(idxNoise, layers_set, noise1_file_idx, delta, L, n, R, Mavail, eps, seedNoise)
            for layer in list(g_1[S[0]].keys()):
                added_noise = noise1_dict[layer][noise1_idx[layer]].reshape(g_1[S[0]][layer].shape).to(args.device).float()
                noise1_idx[layer] += 1
                g_avg_1[layer] = (g_1[S[0]][layer]-g_0[S[0]][layer]) + added_noise
                for idxClient in S[1:]:
                    added_noise = noise1_dict[layer][noise1_idx[layer]].reshape(g_1[idxClient][layer].shape).to(args.device).float()
                    noise1_idx[layer] += 1
                    g_avg_1[layer] += (g_1[idxClient][layer]-g_0[idxClient][layer]) + added_noise
                # Take average of aggregated noisy gradients
                g_avg_1[layer] += g_avg_2[layer]*Mavail
                g_avg_1[layer] /= (Mavail * 1.)
            net_glob_1.zero_grad()
            for layer, param in net_glob_1.named_parameters():
                # print(layer, param.size())
                param.grad = g_avg_1[layer]
            optimizer_1.step()
            net_glob_2 = copy.deepcopy(net_glob_1)

        # if r==0: Lines 4-8 of Algorithm   else: Lines 17-20 of Algorithm
        g_2 = {}  # g_2 = {'layer1': [...], 'layer2': [...]} gradients wrt parameters of each layer
        loss_r_temp = 0 # Sums up loss of clients
        for m in S:
            g_2[m] = {} # gradients of client m
            idxs = np.random.randint(0, train_features[m].shape[0],
                                     K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            loss_total = 0
            for idx in idxs:
                local = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                net_local = copy.deepcopy(net_glob_2)
                _, named_parameters_local, loss = local.train(stepsize, net=net_local.to(args.device))
                # print(loss)
                loss_total += loss
                for layer, param in named_parameters_local:
                    b = copy.deepcopy(param.grad) / (
                                K * 1.)  # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                    if args.clipping != -1:
                        c = min(1, L / torch.norm(b))  # clip
                        b *= c
                    if layer not in g_2[m]:
                        g_2[m][layer] = b
                    else:
                        g_2[m][layer] += b
            loss_total /= (K * 1.)
            loss_r_temp += loss_total
        # take average of gradients
        losses.append(loss_r_temp / Mavail)
        if losses[-1]>losses[0]:
            counter_increase += 1
        print(f"Average loss of clinets from net_glob_2 at round {r}: {losses[-1]}")
        if counter_increase>=50 or losses[-1]>losses[0]*10:
            return losses, 'diverged', net_glob_2

        g_avg_2 = {}
        # d = number of parameters in the layer
        # gauss_AC(d) >> it needs to be reshaped
        for layer in list(g_2[S[0]].keys()):
            idxNoise = 2 # noise type
            # print(noise2_idx[layer], noise_bucket_size)
            if noise2_idx[layer] >= noise_bucket_size:
                noise2_file_idx += 1
                noise2_dict, noise2_idx = load_noise(idxNoise, layers_set, noise2_file_idx, delta, L, n, R, Mavail, eps, seedNoise)
        for layer in list(g_2[S[0]].keys()):
            added_noise = noise2_dict[layer][noise2_idx[layer]].reshape(g_2[S[0]][layer].shape).to(args.device).float()
            noise2_idx[layer] += 1
            g_avg_2[layer] = g_2[S[0]][layer] + added_noise
            # tot_time = time.time() - time_start
            # print(f'Time spent for adding noise to layer {layer} with {x_len} parameters for client 0: {tot_time} seconds')
            for idxClient in S[1:]:
                added_noise = noise2_dict[layer][noise2_idx[layer]].reshape(g_2[idxClient][layer].shape).to(args.device).float()
                noise2_idx[layer] += 1
                g_avg_2[layer] += g_2[idxClient][layer] + added_noise
            # Take average of aggregated noisy gradients
            g_avg_2[layer] /= (Mavail * 1.)
    print('')
    return losses, 'converged', net_glob_2


def mlp_SPIDER(args, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel, avg_window=8):
    np.random.seed(np_seed)
    losses = [] # keeps the loss value at each round
    counter_increase = 0 # Counts the number of times our loss is larger that loss in iteration 1.
    for r in range(0, R):
        #randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        if r==0:
            torch.manual_seed(seedModel) # Seed of model
            net_glob_2 = MLP(dim_in=train_features.shape[-1], dim_hidden=args.dH, dim_out=args.num_classes).to(
                args.device) # Initialize global model
            net_glob_2.train()

            layers_set = set() # Set of layers that model has
            for layer, param in net_glob_2.named_parameters():
                print(f"layer: {layer}, size of parameters: {param.size()}")
                layers_set.add((layer, param.size()))

        elif r>=1:
            net_glob_0 = copy.deepcopy(net_glob_2)
            net_glob_0.train()

            optimizer_2 = torch.optim.SGD(net_glob_2.parameters(), lr=stepsize)
            net_glob_2.zero_grad()
            for layer, param in net_glob_2.named_parameters():
                param.grad = g_avg_2[layer]
            optimizer_2.step()

            net_glob_1 = copy.deepcopy(net_glob_2)
            net_glob_1.train()
            optimizer_1 = torch.optim.SGD(net_glob_1.parameters(), lr=stepsize)

            # Line 16 of Algorithm: Compute gradient w.r.t. w^0
            g_0 = {} # g_0 = {'layer1': [...], 'layer2': [...]}
            g_1 = {} # g_1 = {'layer1': [...], 'layer2': [...]}
            for m in S:
                g_0[m] = {}
                g_1[m] = {}
                idxs = np.random.randint(0, train_features[m].shape[0], K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
                for idx in idxs:
                    local0 = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                    net_local0 = copy.deepcopy(net_glob_0)
                    _, named_parameters_local0, _ = local0.train(stepsize, net=net_local0.to(args.device))
                    for layer, param in named_parameters_local0:
                        b = copy.deepcopy(param.grad)/(K*1.) # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                        if args.clipping!=-1:
                            c = min(1, L / torch.norm(b))  # clip
                            b *= c
                        if layer not in g_0[m]:
                            g_0[m][layer] = b
                        else:
                            g_0[m][layer] += b
                    local1 = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                    net_local1 = copy.deepcopy(net_glob_1)
                    _, named_parameters_local1, _ = local1.train(stepsize, net=net_local1.to(args.device))
                    for layer, param in named_parameters_local1:
                        b = copy.deepcopy(param.grad)/(K*1.) # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                        if args.clipping!=-1:
                            c = min(1, L / torch.norm(b))  # clip
                            b *= c
                        if layer not in g_1[m]:
                            g_1[m][layer] = b
                        else:
                            g_1[m][layer] += b

            # Line 16 of Algorithm: Compute v^1
            g_avg_1 = {}
            # d = number of parameters in the layer
            for layer in list(g_1[S[0]].keys()):
                g_avg_1[layer] = (g_1[S[0]][layer]-g_0[S[0]][layer])
                for idxClient in S[1:]:
                    g_avg_1[layer] += (g_1[idxClient][layer]-g_0[idxClient][layer])
                # Take average of aggregated noisy gradients
                g_avg_1[layer] += g_avg_2[layer]*Mavail
                g_avg_1[layer] /= (Mavail * 1.)
            net_glob_1.zero_grad()
            for layer, param in net_glob_1.named_parameters():
                param.grad = g_avg_1[layer]
            optimizer_1.step()
            net_glob_2 = copy.deepcopy(net_glob_1)

        # if r==0: Lines 4-8 of Algorithm   else: Lines 17-20 of Algorithm
        g_2 = {}  # g_2 = {'layer1': [...], 'layer2': [...]} gradients wrt parameters of each layer
        loss_r_temp = 0 # Sums up loss of clients
        for m in S:
            g_2[m] = {} # gradients of client m
            # g_2 += grad_eval(iterates[-1], K, m) #evaluate stoch grad of log loss at last iterate
            idxs = np.random.randint(0, train_features[m].shape[0],
                                     K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            loss_total = 0
            for idx in idxs:
                local = LocalUpdate(args, train_features[m, idx, :].unsqueeze(0), train_labels[m, idx].unsqueeze(0))
                net_local = copy.deepcopy(net_glob_2)
                _, named_parameters_local, loss = local.train(stepsize, net=net_local.to(args.device))
                loss_total += loss
                for layer, param in named_parameters_local:
                    b = copy.deepcopy(param.grad) / (
                                K * 1.)  # {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}# {0: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}, 1: {'layer1': [gradient of layer1], 'layer2': [gradient of layer2], .....}}
                    if args.clipping != -1:
                        c = min(1, L / torch.norm(b))  # clip
                        b *= c
                    if layer not in g_2[m]:
                        g_2[m][layer] = b
                    else:
                        g_2[m][layer] += b
            loss_total /= (K * 1.)
            loss_r_temp += loss_total
        # take average of gradients
        losses.append(loss_r_temp / Mavail)
        if losses[-1]>losses[0]:
            counter_increase += 1
        print(f"Average loss of clinets from net_glob_2 at round {r}: {losses[-1]}")
        if counter_increase>=50 or losses[-1]>losses[0]*10:
            return losses, 'diverged', net_glob_2

        g_avg_2 = {}
        # d = number of parameters in the layer
        for layer in list(g_2[S[0]].keys()):
            g_avg_2[layer] = g_2[S[0]][layer]
            for idxClient in S[1:]:
                g_avg_2[layer] += g_2[idxClient][layer]
            # Take average of aggregated noisy gradients
            g_avg_2[layer] /= (Mavail * 1.)
    print('')
    return losses, 'converged', net_glob_2

def test_err(net_g, features, labels, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    M = features.shape[0]
    n = features.shape[1]
    for idx_m in range(M):
        for idx_n in range(n):
            data = features[idx_m][idx_n]
            target = labels[idx_m][idx_n]
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            data = data.unsqueeze(0)
            if args.model=="CNN":
                target = target.long()
            elif args.model=="MLP":
                target = target.long().unsqueeze(0)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= (M*n)
    accuracy = 100.00 * correct.item() / (M*n)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, M*n, accuracy))
    return accuracy, test_loss

##################################################################################################################
def main(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    p = 0.0 #for full heterogeneity
    dim = 50
    model = "MLP"
    M = 25 # Number of clients
    path = 'temp'
    n_m = load_MNIST2(p, dim, path, model)[-1] #number of examples (train and test) per digit per machine
    n = int(n_m*2*0.8) #total number of TRAINING examples (two digits) per machine
    DO_COMPUTE = True

    num_trials = 1 #due to time/computation constraints, we recommend keeping num_trials = 1 or 2
    Mavail = 25 # Number of clients
    trainModel = True
    checkAccuracy = True
    # 11: Noisy SPIDER, 12: non-private SPIDER, 21: Noisy MB-SGD, 22: non-private MB-SGD, 31: Noisy local SGD, 32: non-private local SGD
    models2run = [11, 12, 21, 22, 31, 32]
    R = 25
    seedModel = 1
    seedNoise = 1
    epsilons = [0.75, 1, 1.5, 3, 6, 12, 18]
    K_constant = 24
    n_stepsizes = 5
    lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(-9, 0, n_stepsizes)]
    Ls = [1, 5, 10, 100, 10000]
    gEpsStepLproduct = list(itertools.product(epsilons, lg_stepsizes, Ls))
    gStepLproduct = list(itertools.product(lg_stepsizes, Ls))
    delta = 1/(n**2)
    K = int(max(1, n*math.sqrt(K_constant) / (2*math.sqrt(R)))) # needed for privacy by moments acct; 24 >= largest epsilon that we test
    path = 'dp_mnist_p={:.2f}_K={:d}_R={:d}'.format(p,K,R)
    #keep track of train excess risk too
    # Local sgd
    local_test_accuracy = {}
    local_train_accuracy = {}
    local_test_errors = {}
    local_train_errors = {}
    # MB sgd
    MB_test_accuracy = {}
    MB_train_accuracy = {}
    MB_test_errors = {}
    MB_train_errors = {}
    # SPIDER
    SPIDER_test_accuracy = {}
    SPIDER_train_accuracy = {}
    SPIDER_test_errors = {}
    SPIDER_train_errors = {}
    for i, (stepsize, L) in enumerate(gStepLproduct):
        # Local sgd
        local_test_errors[(stepsize, L)] = 0
        local_train_errors[(stepsize, L) ] = 0
        local_test_accuracy[(stepsize, L)] = 0
        local_train_accuracy[(stepsize, L) ] = 0
        # MB sgd
        MB_test_errors[(stepsize, L)] = 0
        MB_train_errors[(stepsize, L) ] = 0
        MB_test_accuracy[(stepsize, L)] = 0
        MB_train_accuracy[(stepsize, L) ] = 0
        # SPIDER
        SPIDER_test_errors[(stepsize, L)] = 0
        SPIDER_train_errors[(stepsize, L) ] = 0
        SPIDER_test_accuracy[(stepsize, L)] = 0
        SPIDER_train_accuracy[(stepsize, L) ] = 0
    # Noisy local sgd
    noisylocal_test_accuracy = {}
    noisylocal_train_accuracy = {}
    noisylocal_test_errors = {}
    noisylocal_train_errors = {}
    # Noisy MB sgd
    noisyMB_test_accuracy = {}
    noisyMB_train_accuracy = {}
    noisyMB_test_errors = {}
    noisyMB_train_errors = {}
    # Noisy SPIDER
    noisySPIDER_test_accuracy = {}
    noisySPIDER_train_accuracy = {}
    noisySPIDER_test_errors = {}
    noisySPIDER_train_errors = {}
    for i, (eps, stepsize, L) in enumerate(gEpsStepLproduct):
        # Noisy local sgd
        noisylocal_test_errors[(eps, stepsize, L)] = 0
        noisylocal_train_errors[(eps, stepsize, L) ] = 0
        noisylocal_test_accuracy[(eps, stepsize, L)] = 0
        noisylocal_train_accuracy[(eps, stepsize, L) ] = 0
        # Noisy MB sgd
        noisyMB_test_errors[(eps, stepsize, L)] = 0
        noisyMB_train_errors[(eps, stepsize, L) ] = 0
        noisyMB_test_accuracy[(eps, stepsize, L)] = 0
        noisyMB_train_accuracy[(eps, stepsize, L) ] = 0
        # Noisy SPIDER
        noisySPIDER_test_errors[(eps, stepsize, L)] = 0
        noisySPIDER_train_errors[(eps, stepsize, L) ] = 0
        noisySPIDER_test_accuracy[(eps, stepsize, L)] = 0
        noisySPIDER_train_accuracy[(eps, stepsize, L) ] = 0
    #keep track of train excess risk too
    # Local sgd
    local_best_test_accuracy = -1
    local_best_train_accuracy = -1
    # MB sgd
    MB_best_test_accuracy = -1
    MB_best_train_accuracy = -1
    # SPIDER
    SPIDER_best_test_accuracy = -1
    SPIDER_best_train_accuracy = -1

    # Noisy local sgd
    noisylocal_best_test_accuracy = {}
    noisylocal_best_train_accuracy = {}
    # Noisy MB sgd
    noisyMB_best_test_accuracy = {}
    noisyMB_best_train_accuracy = {}
    # Noisy SPIDER
    noisySPIDER_best_test_accuracy = {}
    noisySPIDER_best_train_accuracy = {}
    for eps in epsilons:
        # Noisy local sgd
        noisylocal_best_test_accuracy[eps] = -1
        noisylocal_best_train_accuracy[eps] = -1
        # Noisy MB sgd
        noisyMB_best_test_accuracy[eps] = -1
        noisyMB_best_train_accuracy[eps] = -1
        # Noisy SPIDER
        noisySPIDER_best_test_accuracy[eps] = -1
        noisySPIDER_best_train_accuracy[eps] = -1

    if DO_COMPUTE:
        for trial in range(num_trials):
            print("DOING TRIAL", trial)
            train_features, train_labels, test_features, test_labels = load_MNIST2(p,dim,path,model)[0:4]
            train_features = torch.Tensor(train_features)
            train_labels = torch.Tensor(train_labels)
            test_features = torch.Tensor(test_features)
            test_labels = torch.Tensor(test_labels)

            if 11 in models2run and trainModel:
                # Learning noisy SPIDER
                for i, (eps, stepsize, L) in enumerate(gEpsStepLproduct):
                    mlp_noisy_SPIDER_path = os.path.join('models', 'MLP_noisy_SPIDER_New_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_eps={:.2f}_L={:.2g}_stepsize={:.8g}_sM={:.8g}_sN={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,eps,L,stepsize,seedModel,seedNoise))
                    if not os.path.exists(mlp_noisy_SPIDER_path+".pt"):
                        print('Doing noisy MLP SPIDER...') #for each stepsize option, compute average excess risk of LocalSGD over nreps trials
                        print('Epsilon: {:.5f}, Stepsize: {:.5f}, L: {:.5f}'.format(eps, stepsize, L))
                        print('Path: ', mlp_noisy_SPIDER_path)
                        t_now = time.time()
                        l, success, net_noisy_SPIDER = mlp_noisy_SPIDER(args, eps, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel, seedNoise)
                        print(f"Computation time: {(time.time() - t_now) / 60} minute(s)")
                        if success == 'converged':
                            torch.save(net_noisy_SPIDER, mlp_noisy_SPIDER_path+".pt")
                            np.savetxt(mlp_noisy_SPIDER_path+".txt", l)  # write

            if 12 in models2run and trainModel:
                # Learning non-private SPIDER
                for i, (stepsize, L) in enumerate(gStepLproduct):
                    mlp_SPIDER_path = os.path.join('models', 'MLP_SPIDER_New_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_L={:.2g}_stepsize={:.8g}_sM={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,L,stepsize,seedModel))
                    if not os.path.exists(mlp_SPIDER_path+".pt"):
                        print('Doing MLP SPIDER...') #for each stepsize option, compute average excess risk of LocalSGD over nreps trials
                        print('Stepsize: {:.5f}, L: {:.5f}'.format(stepsize, L))
                        print('Path: ', mlp_SPIDER_path)
                        t_now = time.time()
                        l, success, net_SPIDER = mlp_SPIDER(args, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel)
                        print(f"Computation time: {(time.time() - t_now) / 60} minute(s)")
                        if success == 'converged':
                            torch.save(net_SPIDER, mlp_SPIDER_path+".pt")
                            np.savetxt(mlp_SPIDER_path+".txt", l)  # write

            if 21 in models2run and trainModel:
                # Learning noisy MB sgd
                for i, (eps, stepsize, L) in enumerate(gEpsStepLproduct):
                    mlp_noisy_MB_sgd_path = os.path.join('models',
                                                         'MLP_noisy_MiniBatch_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_eps={:.2f}_L={:.2g}_stepsize={:.8g}_sM={:.8g}_sN={:.8g}'.format(
                                                             p,q, K, R,M,Mavail, args.dH, eps, L, stepsize, seedModel, seedNoise))
                    if not os.path.exists(mlp_noisy_MB_sgd_path + ".pt"):
                        print(
                            'Doing noisy MLP MiniBatch SGD...')  # for each stepsize option, compute average excess risk of LocalSGD over nreps trials
                        print('Epsilon: {:.5f}, Stepsize: {:.5f}, L: {:.5f}'.format(eps, stepsize, L))
                        print('Path: ', mlp_noisy_MB_sgd_path)
                        t_now = time.time()
                        l, success, net_noisy_minibatch_sgd = mlp_noisy_minibatch_sgd(args, eps, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel, seedNoise)
                        print(f"Computation time: {(time.time() - t_now) / 60} minute(s)")
                        if success == 'converged':
                            torch.save(net_noisy_minibatch_sgd, mlp_noisy_MB_sgd_path + ".pt")
                            np.savetxt(mlp_noisy_MB_sgd_path + ".txt", l)  # write


            if 22 in models2run and trainModel:
                # Learning MB sgd
                for i, (stepsize, L) in enumerate(gStepLproduct):
                    mlp_MB_sgd_path = os.path.join('models', 'MLP_MB_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_L={:.2g}_stepsize={:.8g}_sM={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,L,stepsize,seedModel))
                    if not os.path.exists(mlp_MB_sgd_path+".pt"):
                        print('Doing MLP MB_sgd...') #for each stepsize option, compute average excess risk of LocalSGD over nreps trials
                        print('Stepsize: {:.5f}, L: {:.5f}'.format(stepsize, L))
                        print('Path: ', mlp_MB_sgd_path)
                        t_now = time.time()
                        l, success, net_MB_sgd = mlp_minibatch_sgd(args, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel)
                        print(f"Computation time: {(time.time() - t_now) / 60} minute(s)")
                        if success == 'converged':
                            torch.save(net_MB_sgd, mlp_MB_sgd_path+".pt")
                            np.savetxt(mlp_MB_sgd_path+".txt", l)  # write


            if 31 in models2run and trainModel:
                # Learning noisy local sgd
                for i, (eps, stepsize, L) in enumerate(gEpsStepLproduct):
                    mlp_noisy_local_sgd_path = os.path.join('models', 'MLP_noisy_local_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_eps={:.2f}_L={:.2g}_stepsize={:.8g}_sM={:.8g}_sN={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,eps,L,stepsize,seedModel,seedNoise))
                    if not os.path.exists(mlp_noisy_local_sgd_path+".pt"):
                        print('Doing noisy MLP local sgd...') #for each stepsize option, compute average excess risk of LocalSGD over nreps trials
                        print('Epsilon: {:.5f}, Stepsize: {:.5f}, L: {:.5f}'.format(eps, stepsize, L))
                        print('Path: ', mlp_noisy_local_sgd_path)
                        t_now = time.time()
                        l, success, net_noisy_local_sgd = mlp_noisy_local_sgd(args, eps, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel)
                        print(f"Computation time: {(time.time() - t_now) / 60} minute(s)")
                        if success == 'converged':
                            torch.save(net_noisy_local_sgd, mlp_noisy_local_sgd_path+".pt")
                            np.savetxt(mlp_noisy_local_sgd_path+".txt", l)  # write


            if 32 in models2run and trainModel:
                # Learning non-private local sgd
                for i, (stepsize, L) in enumerate(gStepLproduct):
                    mlp_local_sgd_path = os.path.join('models', 'MLP_local_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_L={:.2g}_stepsize={:.8g}_sM={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,L,stepsize,seedModel))
                    if not os.path.exists(mlp_local_sgd_path+".pt"):
                        print('Doing MLP local sgd...') #for each stepsize option, compute average excess risk of LocalSGD over nreps trials
                        print('Stepsize: {:.5f}, L: {:.5f}'.format(stepsize, L))
                        print('Path: ', mlp_local_sgd_path)
                        t_now = time.time()
                        l, success, net_local_sgd = mlp_local_sgd(args, delta, n, L, M, Mavail, K, R, stepsize, train_features, train_labels, seedModel)
                        print(f"Computation time: {(time.time() - t_now) / 60} minute(s)")
                        if success == 'converged':
                            torch.save(net_local_sgd, mlp_local_sgd_path+".pt")
                            np.savetxt(mlp_local_sgd_path+".txt", l)  # write


            if 11 in models2run and checkAccuracy:
                # Checking the accuracy of noisy SPIDER
                for i, (eps, stepsize, L) in enumerate(gEpsStepLproduct):
                    mlp_noisy_SPIDER_path = os.path.join('models',
                                                         'MLP_noisy_SPIDER_New_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_eps={:.2f}_L={:.2g}_stepsize={:.8g}_sM={:.8g}_sN={:.8g}'.format(
                                                             p,q, K, R,M,Mavail, args.dH, eps, L, stepsize, seedModel, seedNoise))
                    print('\n', mlp_noisy_SPIDER_path)
                    if os.path.exists(mlp_noisy_SPIDER_path + ".pt"):
                        # print('Loading noisy MLP SPIDER...')
                        # print('Epsilon: {:.5f}, Stepsize: {:.5f}, L: {:.5f}'.format(eps, stepsize, L))
                        net_noisy_SPIDER = torch.load(mlp_noisy_SPIDER_path + ".pt")
                        noisySPIDER_train_accuracy[(eps, stepsize, L)], noisySPIDER_train_errors[(eps, stepsize, L)] = test_err(net_noisy_SPIDER, train_features, train_labels, args)
                        print(f"eps={eps}, noisySPIDER_train_accuracy[(eps, stepsize, L)]: {noisySPIDER_train_accuracy[(eps, stepsize, L)]}, noisySPIDER_test_errors[(eps, stepsize, L)]: {noisySPIDER_test_errors[(eps, stepsize, L)]}")
                        noisySPIDER_test_accuracy[(eps, stepsize, L)], noisySPIDER_test_errors[(eps, stepsize, L)] = test_err(net_noisy_SPIDER, test_features, test_labels, args)
                        print(f"eps={eps}, noisySPIDER_test_accuracy[(eps, stepsize, L)]: {noisySPIDER_test_accuracy[(eps, stepsize, L)]}, noisySPIDER_train_errors[(eps, stepsize, L)]: {noisySPIDER_train_errors[(eps, stepsize, L)]}")


            if 12 in models2run and checkAccuracy:
                # Checking the accuracy of non-private SPIDER
                for i, (stepsize, L) in enumerate(gStepLproduct):
                    mlp_SPIDER_path = os.path.join('models', 'MLP_SPIDER_New_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_L={:.2g}_stepsize={:.8g}_sM={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,L,stepsize,seedModel))
                    print('\n', mlp_SPIDER_path)
                    if os.path.exists(mlp_SPIDER_path + ".pt"):
                        # print('Loading  MLP SPIDER...')
                        # print('Stepsize: {:.5f}, L: {:.5f}'.format(stepsize, L))
                        # load model
                        net_SPIDER = torch.load(mlp_SPIDER_path + ".pt")
                        # train data
                        SPIDER_train_accuracy[(stepsize, L)], SPIDER_train_errors[(stepsize, L)] = test_err(net_SPIDER, train_features, train_labels, args)
                        print(f"SPIDER_train_accuracy[(stepsize, L)]: {SPIDER_train_accuracy[(stepsize, L)]}, SPIDER_test_errors[(stepsize, L)]: {SPIDER_test_errors[(stepsize, L)]}")
                        # test data
                        SPIDER_test_accuracy[(stepsize, L)], SPIDER_test_errors[(stepsize, L)] = test_err(net_SPIDER, test_features, test_labels, args)
                        print(f"SPIDER_test_accuracy[(stepsize, L)]: {SPIDER_test_accuracy[(stepsize, L)]}, SPIDER_train_errors[(stepsize, L)]: {SPIDER_train_errors[(stepsize, L)]}")



            if 21 in models2run and checkAccuracy:
                # Checking the accuracy of noisy MB sgd
                for i, (eps, stepsize, L) in enumerate(gEpsStepLproduct):
                    mlp_noisy_MB_sgd_path = os.path.join('models',
                                                         'MLP_noisy_MiniBatch_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_eps={:.2f}_L={:.2g}_stepsize={:.8g}_sM={:.8g}_sN={:.8g}'.format(
                                                             p,q, K, R,M,Mavail, args.dH, eps, L, stepsize, seedModel, seedNoise))
                    print('\n', mlp_noisy_MB_sgd_path)
                    if os.path.exists(mlp_noisy_MB_sgd_path + ".pt"):
                        # print('Loading noisy MLP MB_sgd...')
                        # print('Epsilon: {:.5f}, Stepsize: {:.5f}, L: {:.5f}'.format(eps, stepsize, L))

                        net_noisy_MB_sgd = torch.load(mlp_noisy_MB_sgd_path + ".pt")

                        noisyMB_train_accuracy[(eps, stepsize, L)], noisyMB_train_errors[(eps, stepsize, L)] = test_err(net_noisy_MB_sgd, train_features, train_labels, args)
                        print(f"eps={eps}, noisyMB_train_accuracy[(eps, stepsize, L)]: {noisyMB_train_accuracy[(eps, stepsize, L)]}, noisyMB_test_errors[(eps, stepsize, L)]: {noisyMB_test_errors[(eps, stepsize, L)]}")

                        noisyMB_test_accuracy[(eps, stepsize, L)], noisyMB_test_errors[(eps, stepsize, L)] = test_err(net_noisy_MB_sgd, test_features, test_labels, args)
                        print(f"eps={eps}, noisyMB_test_accuracy[(eps, stepsize, L)]: {noisyMB_test_accuracy[(eps, stepsize, L)]}, noisyMB_train_errors[(eps, stepsize, L)]: {noisyMB_train_errors[(eps, stepsize, L)]}")




            if 22 in models2run and checkAccuracy:
                # Checking the accuracy of non-private MB sgd
                for i, (stepsize, L) in enumerate(gStepLproduct):
                    mlp_MB_sgd_path = os.path.join('models', 'MLP_MB_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_L={:.2g}_stepsize={:.8g}_sM={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,L,stepsize,seedModel))
                    print('\n', mlp_MB_sgd_path)
                    if os.path.exists(mlp_MB_sgd_path + ".pt"):
                        # print('Loading  MLP MB_sgd...')
                        # print('Stepsize: {:.5f}, L: {:.5f}'.format(stepsize, L))
                        # load model
                        net_MB_sgd = torch.load(mlp_MB_sgd_path + ".pt")
                        # train data
                        MB_train_accuracy[(stepsize, L)], MB_train_errors[(stepsize, L)] = test_err(net_MB_sgd, train_features, train_labels, args)
                        print(f"MB_train_accuracy[(stepsize, L)]: {MB_train_accuracy[(stepsize, L)]}, MB_test_errors[(stepsize, L)]: {MB_test_errors[(stepsize, L)]}")
                        # test data
                        MB_test_accuracy[(stepsize, L)], MB_test_errors[(stepsize, L)] = test_err(net_MB_sgd, test_features, test_labels, args)
                        print(f"MB_test_accuracy[(stepsize, L)]: {MB_test_accuracy[(stepsize, L)]}, MB_train_errors[(stepsize, L)]: {MB_train_errors[(stepsize, L)]}")



            if 31 in models2run and checkAccuracy:
                # Checking the accuracy of noisy local sgd
                for i, (eps, stepsize, L) in enumerate(gEpsStepLproduct):
                    mlp_noisy_local_sgd_path = os.path.join('models',
                                                         'MLP_noisy_local_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_eps={:.2f}_L={:.2g}_stepsize={:.8g}_sM={:.8g}_sN={:.8g}'.format(
                                                             p,q, K, R,M,Mavail, args.dH, eps, L, stepsize, seedModel, seedNoise))
                    print('\n', mlp_noisy_local_sgd_path)
                    if os.path.exists(mlp_noisy_local_sgd_path + ".pt"):
                        try:
                            # print('Loading noisy MLP local sgd...')
                            # print('Epsilon: {:.5f}, Stepsize: {:.5f}, L: {:.5f}'.format(eps, stepsize, L))

                            net_noisy_local_sgd = torch.load(mlp_noisy_local_sgd_path + ".pt")

                            noisylocal_train_accuracy[(eps, stepsize, L)], noisylocal_train_errors[(eps, stepsize, L)] = test_err(net_noisy_local_sgd, train_features, train_labels, args)
                            print(f"eps={eps}, noisylocal_train_accuracy[(eps, stepsize, L)]: {noisylocal_train_accuracy[(eps, stepsize, L)]}, noisylocal_test_errors[(eps, stepsize, L)]: {noisylocal_test_errors[(eps, stepsize, L)]}")

                            noisylocal_test_accuracy[(eps, stepsize, L)], noisylocal_test_errors[(eps, stepsize, L)] = test_err(net_noisy_local_sgd, test_features, test_labels, args)
                            print(f"eps={eps}, noisylocal_test_accuracy[(eps, stepsize, L)]: {noisylocal_test_accuracy[(eps, stepsize, L)]}, noisylocal_train_errors[(eps, stepsize, L)]: {noisylocal_train_errors[(eps, stepsize, L)]}")
                        except:
                            continue


            if 32 in models2run and checkAccuracy:
                # Checking the accuracy of non-private local sgd
                for i, (stepsize, L) in enumerate(gStepLproduct):
                    mlp_local_sgd_path = os.path.join('models', 'MLP_local_sgd_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH={:d}_L={:.2g}_stepsize={:.8g}_sM={:.8g}'.format(p,q,K,R,M,Mavail,args.dH,L,stepsize,seedModel))
                    print('\n', mlp_local_sgd_path)
                    if os.path.exists(mlp_local_sgd_path + ".pt"):
                        try:
                            # print('Loading  MLP local sgd...')
                            # print('Stepsize: {:.5f}, L: {:.5f}'.format(stepsize, L))
                            # load model
                            net_local_sgd = torch.load(mlp_local_sgd_path + ".pt")
                            # train data
                            local_train_accuracy[(stepsize, L)], local_train_errors[(stepsize, L)] = test_err(net_local_sgd, train_features, train_labels, args)
                            print(f"local_train_accuracy[(stepsize, L)]: {local_train_accuracy[(stepsize, L)]}, local_test_errors[(stepsize, L)]: {local_test_errors[(stepsize, L)]}")
                            # test data
                            local_test_accuracy[(stepsize, L)], local_test_errors[(stepsize, L)] = test_err(net_local_sgd, test_features, test_labels, args)
                            print(f"local_test_accuracy[(stepsize, L)]: {local_test_accuracy[(stepsize, L)]}, local_train_errors[(stepsize, L)]: {local_train_errors[(stepsize, L)]}")
                        except:
                            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="MLP", help="Model type")
    parser.add_argument('--gpu', default=-1, type=int, help='if -1 then CPU otherwise GPU')
    parser.add_argument('--clipping', default=1, type=int, help='if -1 then No clipping otherwise DO clipping')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes, MNIST: 2, CIFAR-10: 10")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--dH', type=int, default=64, help='Dimension of hidden layer')
    args = parser.parse_args()
    main(args)
