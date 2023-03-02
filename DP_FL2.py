#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tools import *
import torch
from torch import nn
import argparse
import copy
import time
import itertools
import math
import torch.nn.functional as F
from collections import defaultdict
import matlab.engine
import scipy.io as sio
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, linewidth=240, suppress=True)


# Fix randomness
def fixSeed(seedGlobal):
    np.random.seed(seedGlobal)
    torch.manual_seed(seedGlobal)


# Get noises
def get_iid_gauss_noise(layerSize, L, K, R, delta, n, eps, n_noise, noiseVersion):  # moments account form of noise
    if noiseVersion == 1:
        scaleMultiplier = 32
    elif noiseVersion == 2:
        scaleMultiplier = 8
    std_ = ((scaleMultiplier * (L ** 2) * K * R * (np.log(2 / (delta))) / (n ** 2 * eps ** 2))) ** 0.5
    return torch.randn(n_noise, layerSize) * std_


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


# Create the network
def getNet(args, inputSize=0):
    if args.data == "WBCD":
        return MLPCrossEntropyBC(dim_in=inputSize, dim_hidden=args.dH, dim_out=47).to(args.device)
    if args.data == "CIFAR10":
        return Net_CIFAR10_v1().to(args.device)


def getLayersSet(net):
    layers_set = set()
    for layer, param in net.named_parameters():
        layers_set.add((layer, param.size()))
    return layers_set


class LocalUpdate(object):
    def __init__(self, args, train_features_local, train_labels_local):
        self.args = args
        if self.args.data == "WBCD" or self.args.data == "CIFAR10":
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.features_vectors = train_features_local.to(self.args.device)
        self.labels = train_labels_local.to(self.args.device)


    def forward_input_CNN(self, conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight,
                          fc2_bias, fc3_weight, fc3_bias):
        pool = nn.MaxPool2d(2, 2)

        x = pool(F.relu(F.conv2d(self.features_vectors, weight=conv1_weight, bias=conv1_bias)))
        x = pool(F.relu(F.conv2d(x, weight=conv2_weight, bias=conv2_bias)))
        x = torch.flatten(x, 1)
        x = F.relu(F.linear(x, fc1_weight, fc1_bias))
        x = F.relu(F.linear(x, fc2_weight, fc2_bias))
        x = F.linear(x, fc3_weight, fc3_bias)

        loss = self.loss_func(x, self.labels.long().squeeze(dim=1))
        return loss

    def train(self, net):
        net.train()

        net.zero_grad()
        net_preds = net(self.features_vectors)
        if self.args.data == "WBCD":
            loss = self.loss_func(net_preds, self.labels.long())
        if self.args.data == "CIFAR10":
            loss = self.loss_func(net_preds, self.labels.long().squeeze(dim=1))

        paramTensorsNames = tuple([param[0] for param in net.named_parameters()])
        paramTensors = tuple([param[1] for param in net.named_parameters()])

        if self.args.data == "WBCD":
            grads = torch.autograd.functional.jacobian(self.forward_input_MLP_WBCD, paramTensors, vectorize=True)
        elif self.args.data == "CIFAR10":
            grads = torch.autograd.functional.jacobian(self.forward_input_CNN, paramTensors, vectorize=True)
        return paramTensorsNames, grads, sum(loss).item()

    def train_local_SGD(self, L, stepsize, net, R, eps=0, delta=0, n=0, addNoise=False, noise_dict={}, noise_idx={}):
        net.train()
        # train and update
        losses = []
        optimizer = torch.optim.SGD(net.parameters(), lr=stepsize)
        optimizer = torch.optim.Adam(net.parameters(), lr=stepsize, weight_decay=float(self.args.wD))
        for features_vector, label in zip(self.features_vectors, self.labels):
            features_vector, label = features_vector.to(self.args.device).unsqueeze(0), label.to(
                self.args.device).unsqueeze(0)
            net.zero_grad()
            net_preds = net(features_vector)
            if self.args.data == "WBCD":
                loss = self.loss_func(net_preds, label.long().squeeze(0))
            if self.args.data == "CIFAR10":
                loss = self.loss_func(net_preds, label.long().squeeze(0))
            loss.backward()
            for layer, param in net.named_parameters():
                if args.clipping == 1:
                    c = min(1, L / torch.norm(param.grad))  # clip
                    param.grad.mul_(c)

                if args.clipping == 2:
                    c = min(1, L / torch.norm(param.grad))  # clip
                    param.grad.mul_(c)

                if addNoise:
                    if args.matlabNoise:
                        added_noise = (noise_dict[layer][noise_idx[layer]]).reshape(param.grad.shape).to(args.device).float()
                        noise_idx[layer] += 1
                    else:
                        x_len = np.prod(param.size())
                        added_noise = get_iid_gauss_noise(x_len, L, self.features_vectors.shape[0], R, delta, n, eps, n_noise=1, noiseVersion=2).reshape(
                            param.grad.shape).to(args.device).float()
                    param.grad.add_(added_noise)
            optimizer.step()
            losses.append(loss.item())
        return net.named_parameters(), sum(losses) / len(losses)


def local_sgd(args, L, M, Mavail, K, R, stepsize, train_features, train_labels, eps=0, delta=0, n=0, addNoise=False):
    counter_increase = 0  # Counts the number of times our loss is larger that loss in iteration 1.
    losses = []

    net_glob = getNet(args, train_features.shape[-1])
    net_glob.train()

    if args.matlabNoise:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        layers_set = getLayersSet(net_glob)

    minLoss = float('inf')
    for r in range(R):
        t = time.time()
        # randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        w_end = {}
        for layerName, param in net_glob.named_parameters():
            w_end[layerName] = torch.zeros_like(param)
        loss_m = []
        for m in S:
            net_local = copy.deepcopy(net_glob)
            idxs = np.random.randint(0, train_features[m].shape[0], K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            local = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
            if args.matlabNoise:
                noise_dict = {}
                noise_idx = {}
                for layer, param_size in layers_set:
                    x_len = np.prod(param_size)
                    seedNoiseTemp = np.random.rand(1)
                    noise_dict[layer] = torch.tensor(eng.noise_generator_localSGD2(float(x_len), float(eps), float(delta), float(n), float(R), float(L), float(K), float(seedNoiseTemp), float(K), nargout=1))
                    noise_idx[layer] = 0
                named_parameters_local, loss = local.train_local_SGD(L, stepsize, net=net_local.to(args.device), R=R,
                                                                     eps=eps, delta=delta, n=n, addNoise=addNoise, noise_dict=noise_dict, noise_idx=noise_idx)
            else:
                named_parameters_local, loss = local.train_local_SGD(L, stepsize, net=net_local.to(args.device), R=R,
                                                                     eps=eps, delta=delta, n=n, addNoise=addNoise)

            loss_m.append(loss)
            for layerName, param in named_parameters_local:
                w_end[layerName].add_((param.detach().clone()) / Mavail)  # NOISE IS NOT FOR WEIGHT >> NOISE is for gradient of the local machine during training
        # take average of gradients
        losses.append(sum(loss_m) / Mavail)
        print(f"Loss at round {r}: {losses[-1]}")

        if losses[-1] > losses[0]:
            counter_increase += 1

        if math.isnan(losses[-1]):
            return losses, 'diverged', net_glob
        if counter_increase >= 50 or losses[-1] > losses[0] * 10:
            return losses, 'diverged', net_glob

        with torch.no_grad():
            for layerName, param in net_glob.named_parameters():
                param.copy_(w_end[layerName].detach().clone())

        if losses[-1] <= minLoss or r == 0:
            minLoss = losses[-1]
            best_agent = copy.deepcopy(net_glob)

    if args.matlabNoise:
        # Stop engine
        eng.quit()
    return losses, 'converged', best_agent


def clip_gradient(args, paramTensorsNames, grads, g, L):
    LDictClient = {}
    for layerName, grad in zip(paramTensorsNames, grads):
        b = copy.deepcopy(grad).to(args.device)
        bNorm = torch.norm(torch.flatten(b, start_dim=1, end_dim=-1), dim=1)
        if args.clipping == 1:
            LDictClient[layerName] = L
        if args.clipping == 2:
            LDictClient[layerName] = torch.median(bNorm).item()
        c = torch.min(torch.ones(bNorm.shape).to(args.device), LDictClient[layerName]/bNorm)
        cReshape = tuple([b.shape[0]] + [1] * (len(b.shape) - 1))
        b.mul_(c.reshape(cReshape))
        assert layerName not in g, f"Check dictionary g. {layerName} is already in the dictionary!"
        g[layerName] = b.mean(dim=0)
    return g, LDictClient


def minibatch_sgd(args, L, M, Mavail, K, R, stepsize, train_features, train_labels, eps=0, delta=0, n=0,
                  addNoise=False):
    counter_increase = 0  # Counts the number of times our loss is larger that loss in iteration 1.

    net_glob = getNet(args, train_features.shape[-1])
    net_glob.train()

    optimizer = torch.optim.SGD(net_glob.parameters(), lr=stepsize)

    losses = []

    layers_set = getLayersSet(net_glob)

    if args.matlabNoise:
        noise_file_idx = 1
        if R%Mavail==0:
            noise_bucket_size = R
        else:
            noise_bucket_size = Mavail
        assert noise_bucket_size%Mavail == 0
        noise_dict, noise_idx = load_noise(2, layers_set, noise_file_idx, delta, L, n, R, Mavail, eps, args.seedModel)

    minLoss = float('inf')
    for r in range(R):
        t_round = time.time()
        # randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        g = {}
        LDict = {}
        loss_r_temp = 0
        # t = time.time()
        for m in S:
            g[m] = {}
            idxs = np.random.randint(0, train_features[m].shape[0],
                                     K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
            local = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
            net_local = copy.deepcopy(net_glob)
            paramTensorsNames, grads, loss_total = local.train(net=net_local.to(args.device))
            g[m], LDict[m] = clip_gradient(args, paramTensorsNames, grads, g[m], L)
            loss_total /= (K * 1.)
            loss_r_temp += loss_total
        # take average of gradients
        losses.append(loss_r_temp / Mavail)
        print(f"Loss at round {r}: {losses[-1]}")
        if losses[-1] > losses[0]:
            counter_increase += 1
            
        if math.isnan(losses[-1]):
            return losses, 'diverged', net_glob
        if counter_increase >= 50 or losses[-1] > losses[0] * 10:
            return losses, 'diverged', net_glob

        g_avg = {}

        if args.matlabNoise:
            for layer, _ in layers_set:
                if noise_idx[layer] >= noise_bucket_size:
                    noise_file_idx += 1
                    noise_dict, noise_idx = load_noise(2, layers_set, noise_file_idx, delta, L, n, R, Mavail, eps, args.seedModel)

        for layer, param_size in layers_set:
            x_len = np.prod(param_size)
            t_now = time.time()
            if addNoise:
                if args.matlabNoise:
                    added_noise = noise_dict[layer][noise_idx[layer]].reshape(g[S[0]][layer].shape).to(args.device).float()
                    noise_idx[layer] += 1
                else:
                    added_noise = get_iid_gauss_noise(x_len, LDict[S[0]][layer], 1, R, delta, n, eps, n_noise=1, noiseVersion=2).reshape(
                        g[S[0]][layer].shape).to(args.device).float()
            else:
                added_noise = 0
            g_avg[layer] = g[S[0]][layer] + added_noise
            for idxClient in S[1:]:
                time_start = time.time()
                if addNoise:
                    if args.matlabNoise:
                        added_noise = noise_dict[layer][noise_idx[layer]].reshape(g[idxClient][layer].shape).to(args.device).float()
                        noise_idx[layer] += 1
                    else:
                        added_noise = get_iid_gauss_noise(x_len, LDict[idxClient][layer], 1, R, delta, n, eps, n_noise=1, noiseVersion=2).reshape(
                            g[idxClient][layer].shape).to(args.device).float()
                else:
                    added_noise = 0
                g_avg[layer].add_(g[idxClient][layer] + added_noise)
            # Take average of aggregated noisy gradients
            g_avg[layer] /= (Mavail * 1.)

        net_glob.zero_grad()
        for layer, param in net_glob.named_parameters():
            param.grad = g_avg[layer].detach().clone()
        optimizer.step()

        if losses[-1] <= minLoss or r == 0:
            minLoss = losses[-1]
            best_agent = copy.deepcopy(net_glob)
        print(f"Round {r} computation time: {((time.time() - t_round) / 60):.4g} minutes")
    return losses, 'converged', best_agent


def spider(args, L, M, Mavail, K, R, stepsize, train_features, train_labels, eps=0, delta=0, n=0, addNoise=False):
    losses = []  # keeps the loss value at each round
    counter_increase = 0  # Counts the number of times our loss is larger that loss in iteration 1.
    if args.matlabNoise:
        noise1_file_idx = 1 # Index of the bucket of file of noise 1
        noise2_file_idx = 1 # Index of the bucket of file of noise 2
        if R%Mavail==0:
            noise_bucket_size = R
        else:
            noise_bucket_size = Mavail
        assert noise_bucket_size%Mavail == 0
    minLoss = float('inf')
    for r in range(0, R):
        t_round = time.time()
        # randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        if r == 0:
            net_glob_2 = getNet(args, train_features.shape[-1])
            net_glob_2.train()
            layers_set = getLayersSet(net_glob_2)
            if args.matlabNoise:
                noise1_dict, noise1_idx = load_noise(1, layers_set, noise1_file_idx, delta, L, n, R, Mavail, eps, args.seedModel) # Load first bucket of file of noise 1
                noise2_dict, noise2_idx = load_noise(2, layers_set, noise2_file_idx, delta, L, n, R, Mavail, eps, args.seedModel) # Load first bucket of file of noise 2

        elif r >= 1:
            net_glob_0 = copy.deepcopy(net_glob_2)
            net_glob_0.train()
            optimizer_2 = torch.optim.SGD(net_glob_2.parameters(), lr=stepsize)
            net_glob_2.zero_grad()
            for layer, param in net_glob_2.named_parameters():
                param.grad = g_avg_2[layer].detach().clone()
            optimizer_2.step()
            net_glob_1 = copy.deepcopy(net_glob_2)
            net_glob_1.train()
            optimizer_1 = torch.optim.SGD(net_glob_1.parameters(), lr=stepsize)
            # Line 16 of Algorithm: Compute gradient w.r.t. w^0
            g_0 = {}  # g_0 = {'layer1': [...], 'layer2': [...]}
            g_1 = {}  # g_1 = {'layer1': [...], 'layer2': [...]}
            LDict0 = {}
            LDict1 = {}
            for m in S:
                g_0[m] = {}
                g_1[m] = {}
                idxs = np.random.randint(0, train_features[m].shape[0],
                                         K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset

                local0 = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
                net_local0 = copy.deepcopy(net_glob_0)
                paramTensorsNames, grads0, _ = local0.train(net=net_local0.to(args.device))
                g_0[m], LDict0[m] = clip_gradient(args, paramTensorsNames, grads0, g_0[m], L)

                local1 = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
                net_local1 = copy.deepcopy(net_glob_1)
                paramTensorsNames, grads1, _ = local1.train(net=net_local1.to(args.device))
                g_1[m], LDict1[m] = clip_gradient(args, paramTensorsNames, grads1, g_1[m], L)
            g_avg_1 = {}
            if args.matlabNoise:            
                for layer in list(g_1[S[0]].keys()):
                    idxNoise = 1
                    if noise1_idx[layer] >= noise_bucket_size:
                        noise1_file_idx += 1
                        noise1_dict, noise1_idx = load_noise(idxNoise, layers_set, noise1_file_idx, delta, L, n, R, Mavail, eps, args.seedModel)

            for layer, param_size in layers_set:
                x_len = np.prod(param_size)
                if addNoise:
                    if args.matlabNoise:
                        added_noise = noise1_dict[layer][noise1_idx[layer]].reshape(g_1[S[0]][layer].shape).to(args.device).float()
                        noise1_idx[layer] += 1
                    else:
                        added_noise = get_iid_gauss_noise(x_len, max(LDict0[S[0]][layer], LDict1[S[0]][layer]), 1, R, delta, n, eps, n_noise=1, noiseVersion=1).reshape(
                            g_1[S[0]][layer].shape).to(args.device).float()
                else:
                    added_noise = 0
                g_avg_1[layer] = (g_1[S[0]][layer] - g_0[S[0]][layer]) + added_noise
                for idxClient in S[1:]:
                    if addNoise:
                        if args.matlabNoise:
                            added_noise = noise1_dict[layer][noise1_idx[layer]].reshape(g_1[idxClient][layer].shape).to(args.device).float()
                            noise1_idx[layer] += 1
                        else:
                            added_noise = get_iid_gauss_noise(x_len, max(LDict0[idxClient][layer], LDict1[idxClient][layer]), 1, R, delta, n, eps, n_noise=1,
                                                              noiseVersion=1).reshape(g_1[idxClient][layer].shape).to(args.device).float()
                    else:
                        added_noise = 0
                    g_avg_1[layer].add_((g_1[idxClient][layer] - g_0[idxClient][layer]) + added_noise)
                # Take average of aggregated noisy gradients
                g_avg_1[layer].add_(g_avg_2[layer] * Mavail)
                g_avg_1[layer] /= (Mavail * 1.)
            net_glob_1.zero_grad()
            for layer, param in net_glob_1.named_parameters():
                param.grad = g_avg_1[layer].detach().clone()
            optimizer_1.step()
            net_glob_2 = copy.deepcopy(net_glob_1)
        LDict2 = {}
        g_2 = {}  
        loss_r_temp = 0  # Sums up loss of clients
        for m in S:
            g_2[m] = {}  # gradients of client m

            idxs = np.random.randint(0, train_features[m].shape[0],
                                     K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset

            local = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
            net_local = copy.deepcopy(net_glob_2)
            paramTensorsNames, grads, loss_total = local.train(net=net_local.to(args.device))
            g_2[m], LDict2[m] = clip_gradient(args, paramTensorsNames, grads, g_2[m], L)

            loss_total /= (K * 1.)
            loss_r_temp += loss_total
        # take average of gradients
        losses.append(loss_r_temp / Mavail)
        if losses[-1] > losses[0]:
            counter_increase += 1
        print(f"Average loss of clinets from net_glob_2 at round {r}: {losses[-1]}")
        if math.isnan(losses[-1]):
            return losses, 'diverged', net_glob_2
        if counter_increase >= 50 or losses[-1] > losses[0] * 10:
            return losses, 'diverged', net_glob_2

        if args.matlabNoise:
            for layer in list(g_2[S[0]].keys()):
                idxNoise = 2 # noise type
                if noise2_idx[layer] >= noise_bucket_size:
                    noise2_file_idx += 1
                    noise2_dict, noise2_idx = load_noise(idxNoise, layers_set, noise2_file_idx, delta, L, n, R, Mavail, eps, args.seedModel)
        
        g_avg_2 = {}

        for layer, param_size in layers_set:
            x_len = np.prod(param_size)
            if addNoise:
                if args.matlabNoise:
                    added_noise = noise2_dict[layer][noise2_idx[layer]].reshape(g_2[S[0]][layer].shape).to(args.device).float()
                    noise2_idx[layer] += 1
                else:
                    added_noise = get_iid_gauss_noise(x_len, LDict2[S[0]][layer], 1, R, delta, n, eps, n_noise=1, noiseVersion=2).reshape(
                        g_2[S[0]][layer].shape).to(args.device).float()
            else:
                added_noise = 0
            g_avg_2[layer] = g_2[S[0]][layer] + added_noise
            for idxClient in S[1:]:
                if addNoise:
                    if args.matlabNoise:
                        added_noise = noise2_dict[layer][noise2_idx[layer]].reshape(g_2[idxClient][layer].shape).to(args.device).float()
                        noise2_idx[layer] += 1
                    else:
                        added_noise = get_iid_gauss_noise(x_len, LDict2[idxClient][layer], 1, R, delta, n, eps, n_noise=1, noiseVersion=2).reshape(
                            g_2[idxClient][layer].shape).to(args.device).float()
                else:
                    added_noise = 0
                g_avg_2[layer].add_(g_2[idxClient][layer] + added_noise)
            # Take average of aggregated noisy gradients
            g_avg_2[layer] /= (Mavail * 1.)

        if losses[-1] <= minLoss or r == 0:
            minLoss = losses[-1]
            best_agent = copy.deepcopy(net_glob_2)

        print(f"Round {r} computation time: {((time.time() - t_round) / 60):.4g} minutes
    return losses, 'converged', best_agent


def get_iid_gauss_noise_spider_boost(layerSize, L, R, delta, n, eps, n_noise,
                                     noiseVersion):  # moments account form of noise
    if noiseVersion == 1:
        scaleMultiplier = 16
        std_ = ((scaleMultiplier * (L ** 2) * (np.log(1 / (delta))) / (
                    n ** 2 * eps ** 2))) ** 0.5  # max(R/q, 1) is always equal to R/q because R>=q
    elif noiseVersion == 2:
        scaleMultiplier = 64
        std_ = (scaleMultiplier * (L ** 2) * R * (np.log(1 / (delta))) / ((n ** 2) * (eps ** 2))) ** 0.5
    return torch.randn(n_noise, layerSize) * std_


def spider_boost(args, L, M, Mavail, K, R, stepsize, train_features, train_labels, eps=0, delta=0, n=0, addNoise=False,
                 qR=1):
    K_constant = 24
    K1 = int(max(1, n * math.sqrt(K_constant) / 2 * (min(1, math.sqrt(qR / R)))))  # needed for privacy by advanced comp; 24 = largest epsilon that we test
    K2 = int(max(1, n * math.sqrt(K_constant) / (2 * math.sqrt(R))))  # needed for privacy by advanced comp; 24 = largest epsilon that we test
    K = max(K, K1, K2)
    print(f'Spider boost K: {K}')
    # t = time.time()
    counter_increase = 0  # Counts the number of times our loss is larger that loss in iteration 1.
    net_glob = getNet(args, train_features.shape[-1])
    net_glob.train()
    # optimizer = torch.optim.SGD(net_glob.parameters(), lr=stepsize)
    losses = []
    layers_set = getLayersSet(net_glob)
    minLoss = float('inf')

    for r in range(R):
        LDictPrev = {}
        LDict = {}
        t_round = time.time()
        # randomly choose Mavail out of the Mavail clients:
        S = np.random.choice(list(range(M)), size=Mavail, replace=False, p=None)
        loss_r_temp = 0
        g_prev = defaultdict(lambda: {})
        g = defaultdict(lambda: {})
        for m in S:
            if np.mod(r, qR) == 0:
                idxs = np.random.randint(0, train_features[m].shape[0],
                                         K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
                local = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
                net_local = copy.deepcopy(net_glob)
                paramTensorsNames, grads, loss_total = local.train(net=net_local.to(args.device))
                loss_r_temp += (loss_total / (K * 1.))
            else:
                idxs = np.random.randint(0, train_features[m].shape[0],
                                         K)  # draw minibatch (unif w/replacement) index set of size minibatch_size from machine m's dataset
                local1 = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
                net_local1 = copy.deepcopy(net_glob_prev)
                paramTensorsNames1, grads1, loss_total1 = local1.train(net=net_local1.to(args.device))
                local2 = LocalUpdate(args, train_features[m, idxs, :], train_labels[m, idxs])
                net_local2 = copy.deepcopy(net_glob)
                paramTensorsNames2, grads2, loss_total2 = local2.train(net=net_local2.to(args.device))
                g[m], LDict[m] = clip_gradient(args, paramTensorsNames2, grads2, g[m], L)
                loss_r_temp += (loss_total1 + loss_total2) / (2 * K * 1.)

        net_glob_prev = copy.deepcopy(net_glob)
        net_glob_prev.zero_grad()
        # Taking average of gradients
        # t = time.time()
        g_avg = {}
        for layer, param_size in layers_set:
            x_len = np.prod(param_size)
            t_now = time.time()
            added_noise = 0
            if addNoise:
                if np.mod(r, qR) == 0:  # layerSize, L, R, delta, n, eps, n_noise, noiseVersion)
                    added_noise = get_iid_gauss_noise_spider_boost(x_len, LDict[S[0]][layer], R, delta, n, eps, n_noise=1,
                                                                   noiseVersion=1).reshape(g[S[0]][layer].shape).to(args.device).float()

                else:
                    added_noise = get_iid_gauss_noise_spider_boost(x_len, max(LDict[S[0]][layer], LDictPrev[S[0]]), R, delta, n, eps, n_noise=1,
                                                                   noiseVersion=2).reshape(g[S[0]][layer].shape).to(args.device).float()
            g_avg[layer] = g[S[0]][layer] + added_noise
            if np.mod(r, qR) != 0:
                g_avg[layer].sub_(g_prev[S[0]][layer])

            for idxClient in S[1:]:
                time_start = time.time()
                added_noise = 0
                if addNoise:
                    if np.mod(r, qR) == 0:
                        added_noise = get_iid_gauss_noise_spider_boost(x_len, LDict[idxClient][layer], R, delta, n, eps, n_noise=1,
                                                                       noiseVersion=1).reshape(g[idxClient][layer].shape).to(args.device).float()
                    else:
                        added_noise = get_iid_gauss_noise_spider_boost(x_len, max(LDict[idxClient][layer], LDictPrev[idxClient][layer]), R, delta, n, eps, n_noise=1,
                                                                       noiseVersion=2).reshape(g[idxClient][layer].shape).to(args.device).float()
                g_avg[layer].add_(g[idxClient][layer] + added_noise)
                if np.mod(r, qR) != 0:
                    g_avg[layer].sub_(g_prev[idxClient][layer])
        # Take average of aggregated noisy gradients
        for layer, param_size in layers_set:
            g_avg[layer].div_(Mavail * 1.)
            if np.mod(r, qR) != 0:
                g_avg[layer].add_(g_avg_prev[layer])

        losses.append(loss_r_temp / Mavail)
        g_avg_prev = copy.deepcopy(g_avg)
        net_glob.zero_grad()
        with torch.no_grad():
            for layer, param in net_glob.named_parameters():
                param.sub_(stepsize * g_avg[layer])

        if losses[-1] <= minLoss or r == 0:
            minLoss = losses[-1]
            best_agent = copy.deepcopy(net_glob)
        print(f"Loss at round {r}: {losses[-1]}")
        print(f"Round {r} computation time: {((time.time() - t_round) / 60):.4g} minutes")
        if losses[-1] > losses[0]:
            counter_increase += 1
        if math.isnan(losses[-1]):
            return losses, 'diverged', net_glob
        if counter_increase >= 25 or losses[-1] > losses[0] * 2:
            return losses, 'diverged', net_glob
    return losses, 'converged', best_agent


def pltLoss(losses):
    plt.plot(losses)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.show()


def test_err_cls(net_g, features_vectors, labels, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    M = features_vectors.shape[0]
    n = features_vectors.shape[1]
    for idx_m in range(M):
        for idx_n in range(n):
            data = features_vectors[idx_m][idx_n]
            target = labels[idx_m][idx_n]
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            data = data.unsqueeze(0)
            if args.data == "WBCD":
                target = target.long().unsqueeze(0)
            if args.data == "CIFAR10":
                target = target.long()

            net_preds = net_g(data)
            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum')
            test_loss += loss_func(net_preds, target).item()
            # get the index of the max log-probability
            y_pred = net_preds.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= (M * n)
    accuracy = 100.00 * correct.item() / (M * n)
    return [accuracy, test_loss]


def saveNet(net, path, l):
    torch.save(net, path + ".pt")
    np.savetxt(path + ".txt", l)


def getAlgoCode(algorithmTemp):
    algCodeDict = {'noisy_spider': 11, 'spider': 12, 'noisy_mb_sgd': 21, 'mb_sgd': 22, 'noisy_local_sgd': 31,
                   'local_sgd': 32, 'noisy_spider_boost': 41, 'spider_boost': 42, 'noisy_FedAvg': 51, 'FedAvg': 52}
    return algCodeDict[algorithmTemp]


def getAlgoName(modelIdx):
    algNameDict = {11: 'noisy_spider', 12: 'spider', 21: 'noisy_mb_sgd', 22: 'mb_sgd', 31: 'noisy_local_sgd',
                   32: 'local_sgd', 41: 'noisy_spider_boost', 42: 'spider_boost', 51: 'noisy_FedAvg', 52: 'FedAvg'}
    return algNameDict[modelIdx]


def getAlgoFunction(modelIdx):
    algFunctionDict = {11: spider, 12: spider, 21: minibatch_sgd, 22: minibatch_sgd, 31: local_sgd, 32: local_sgd,
                       41: spider_boost, 42: spider_boost, 51: fedAvg, 52: fedAvg}
    return algFunctionDict[modelIdx]

def getAddress(args, epsilon, modelIdx, K, L, stepsize, stepsize_local, seedData, qR=0):
    alg_name = getAlgoName(modelIdx)

    if args.data == "WBCD":
        return 'MLPCrossEntropy_' + alg_name + '_WBCD_v7_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_dH1={:d}_dH2={:d}_eps={:.2f}_L={:.8g}_stepsize={:.8g}_stepsizeL={:.8g}_sM={:.8g}_sD={:.8g}_nSp={:.8g}_nSt={:.8g}_nStL={:.8g}_qR={:.8g}'.format(
            args.p, args.q_pct_of_data, K, args.R, args.M, args.Mavail, args.dH, args.dH2, epsilon, L, stepsize,
            stepsize_local, args.seedModel, seedData, args.nSplit, args.nStepsize, args.nStepsizeL, qR)
    if args.data == "CIFAR10":
        return 'CIFAR10_' + alg_name + '_v7_p={:.2f}_q={:g}_K={:d}_R={:d}_M={:d}_Mavail={:d}_eps={:.2f}_L={:.8g}_stepsize={:.8g}_stepsizeL={:.8g}_sM={:.8g}_sD={:.8g}_nSt={:.8g}_nStL={:.8g}_qR={:.8g}'.format(
            args.p, args.q_pct_of_data, K, args.R, args.M, args.Mavail, epsilon, L, stepsize, stepsize_local, args.seedModel, seedData,
            args.nStepsize, args.nStepsizeL, qR)

def getSettings(args):
    settings = []
    lg_stepsizes = [np.exp(exponent) for exponent in np.linspace(args.minStepsize, args.maxStepsize, args.nStepsize)]
    lg_stepsizes_local = [np.exp(exponent) for exponent in np.linspace(args.minStepsizeL, args.maxStepsizeL, args.nStepsizeL)]

    if args.clipping==2:
        args.Ls = [0]

    for modelIdx in args.models2run:
        if modelIdx in [11, 21, 31, 51]:
            settings += list(itertools.product([modelIdx], args.epsilons, lg_stepsizes, lg_stepsizes_local, args.Ls, [0]))
        elif modelIdx in [12, 22, 32, 52]:
            settings += list(itertools.product([modelIdx], [1000], lg_stepsizes, lg_stepsizes_local, args.Ls, [0]))
        elif modelIdx in [41]:
            settings += list(itertools.product([modelIdx], args.epsilons, lg_stepsizes, lg_stepsizes_local, args.Ls, args.qRList))
        elif modelIdx in [42]:
            settings += list(itertools.product([modelIdx], [1000], lg_stepsizes, lg_stepsizes_local, args.Ls, args.qRList))
    return settings


def trainModel(args, settings, K, seedData, delta, n, train_features, train_labels):
    settingsLoop = settings

    checked_trainings = set()
    for (modelIdx, eps, stepsize, stepsize_local, L, qR) in settingsLoop:
        fixSeed(seedData)
        algName = getAlgoName(modelIdx)
        algoFunction = getAlgoFunction(modelIdx)
        if modelIdx not in [51, 52]:
            stepsize_local = 0
        path_temp = getAddress(args, eps, modelIdx, K, L, stepsize, stepsize_local, seedData, qR)
        model_path = os.path.join('models', path_temp)
        if not os.path.exists(model_path + ".pt") and (model_path not in checked_trainings):
            checked_trainings.add(model_path)
            print('Algorithm: {:s},\tEpsilon: {:.5f}, Stepsize: {:.5f}, Stepsize local: {:.5f}, L: {:.5f}, qR: {:.5f}'.format(algName, eps,
                                                                                                      stepsize, stepsize_local, L, qR))
            t_now = time.time()
            if modelIdx in [11, 21, 31]:
                l, success, net = algoFunction(args, L, args.M, args.Mavail, K, args.R, stepsize, train_features,
                                               train_labels, eps, delta, n, addNoise=True)
            elif modelIdx in [12, 22, 32]:
                l, success, net = algoFunction(args, L, args.M, args.Mavail, K, args.R, stepsize, train_features, train_labels)
            if modelIdx in [51]:
                l, success, net = algoFunction(args, L, args.M, args.Mavail, K, args.R, stepsize, stepsize_local, train_features,
                                               train_labels, eps, delta, n, addNoise=True)
            elif modelIdx in [52]:
                l, success, net = algoFunction(args, L, args.M, args.Mavail, K, args.R, stepsize, stepsize_local, train_features, train_labels)
            elif modelIdx in [41]:
                l, success, net = algoFunction(args, L, args.M, args.Mavail, K, args.R, stepsize, train_features,
                                               train_labels, eps, delta, n, addNoise=True, qR=qR)
            elif modelIdx in [42]:
                l, success, net = algoFunction(args, L, args.M, args.Mavail, K, args.R, stepsize, train_features, train_labels, qR=qR)
            print(f"Computation time of {algName}: {((time.time() - t_now) / 60):.4g} minute(s)")
            if success == 'converged': saveNet(net, model_path, l)


def testModel(args, settings, K, seedData, train_features, train_labels, test_features, test_labels, test_err, dfResult):
    if args.saveSummary:
        settingsLoop = settings
    else:
        settingsLoop = settings
    checked_accuracies = set()
    for (modelIdx, eps, stepsize, stepsize_local, L, qR) in settingsLoop:
        if modelIdx in [11, 21, 12, 22, 41, 42]:
            stepsize_local = 0
        print("Setting: [seedData, modelIdx, eps, stepsize, stepsize_local, L, qR] ", seedData, modelIdx, eps, stepsize, stepsize_local, L, qR)
        algName = getAlgoName(modelIdx)
        path_temp = getAddress(args, eps, modelIdx, K, L, stepsize, stepsize_local, seedData, qR)
        model_path = os.path.join('models', path_temp)
        if (os.path.exists(model_path + ".pt")) and (model_path not in checked_accuracies):
            checked_accuracies.add(model_path)
            print('\n', model_path)
            net = torch.load(model_path + ".pt", map_location=args.device)
            train_accuracy, train_errors = test_err(net, train_features, train_labels, args)
            test_accuracy, test_errors = test_err(net, test_features, test_labels, args)
            print(f"{algName}, train data, eps: {eps}, stepsize: {stepsize}, stepsize_local: {stepsize_local}, L: {L}, qR: {qR}, \n\tAccuracy: {train_accuracy}")
            print(f"{algName}, test data, eps: {eps}, stepsize: {stepsize}, stepsize_local: {stepsize_local}, L: {L}, qR: {qR}, , \n\t Accuracy: {test_accuracy}")

            dfResultTrain = {'algorithm': algName, 'data': args.data, 'train_test': 'train', 'eps': eps,
                             'stepsize': stepsize, 'stepsize_local': stepsize_local, 'L': L, 'qR': qR, 'seedData': seedData, 'accuracy': train_accuracy}
            dfResult = dfResult.append(dfResultTrain, ignore_index=True)
            dfResultTest = {'algorithm': algName, 'data': args.data, 'train_test': 'test', 'eps': eps,
                            'stepsize': stepsize, 'stepsize_local': stepsize_local, 'L': L, 'qR': qR, 'seedData': seedData, 'accuracy': test_accuracy}
            dfResult = dfResult.append(dfResultTest, ignore_index=True)
    return dfResult

              
def main(args, seed):
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    args.seedModel = copy.deepcopy(seed)
    seedData = copy.deepcopy(seed)
    if args.data == "CIFAR10":
        train_features, train_labels, test_features, test_labels, n_m = load_cifar10_data(args, seedData,
                                                                                          doShuffle=True)  # number of examples per label per machine
        test_err = test_err_cls
        n = n_m  # total number of TRAINING examples per machine
    if args.data == 'WBCD':        
        for seedTemp in range(args.nSeedData):
            _, _, _, _, n_m_K_Temp = load_wbcd(p=args.p, seedData=seedTemp, q_pct_of_data=args.q_pct_of_data, nTask=2, seedSplit=seedTemp, nSplit=args.nSplit, testSplitIdx=0)  # number of examples per label per machine
            n_m_K = min(n_m_K, n_m_K_Temp)
        train_features, train_labels, test_features, test_labels, n_m_original = load_wbcd(p=args.p, seedData=seedData, q_pct_of_data=args.q_pct_of_data, nTask=2, seedSplit=seedData, nSplit=args.nSplit, testSplitIdx=0)  # number of examples per label per machine
        test_err = test_err_cls
        n = int(n_m_K * (1 - 1 / args.nSplit))  # total number of TRAINING examples per machine

    train_features, train_labels = torch.Tensor(train_features), torch.Tensor(train_labels)
    test_features, test_labels = torch.Tensor(test_features), torch.Tensor(test_labels)
    print(f"Train data shape (features, labels):  ({train_features.shape}, {train_labels.shape})")
    print(f"Test data shape (features, labels):  ({test_features.shape}, {test_labels.shape})")
    print(f"Number of examples per machine: {n}")

    K = int(max(1, n * math.sqrt(args.K_constant) / (2 * math.sqrt(args.R))))  # needed for privacy by advanced comp; 24 = largest epsilon that we test

    delta = 1 / (n ** 2)

    settings = getSettings(args)
    print(f'Length of settings: {len(settings)}')
    trainModel(args, settings, K, seedData, delta, n, train_features, train_labels)

    dfResult = pd.DataFrame(columns=['algorithm', 'train_test', 'eps', 'stepsize', 'stepsize_local', 'L'])
    dfResult = testModel(args, settings, K, seedData, train_features, train_labels, test_features, test_labels, test_err, dfResult)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=10, type=int)  
    parser.add_argument('--gpu', default=9, type=int, help='if -1 then CPU otherwise GPU')
    parser.add_argument('--clipping', default=1, type=int, help='1: Grid search clipping threshold. 2: Heuristic clipping threshold. OW: no clipping')
    parser.add_argument('--matlabNoise', type=bool, default=False)
    parser.add_argument('--dH', type=int, default=64, help='Dimension of hidden layer 1')
    parser.add_argument('--dH2', type=int, default=-1, help='Dimension of hidden layer 2')
    parser.add_argument('--data', type=str, default="CIFAR10", help='Data', choices=['CIFAR10', 'WBCD'])
    parser.add_argument('--nSplit', type=int, default=5, help='Number of splits of data')
    parser.add_argument('--nStepsize', type=int, default=12, help='Number of stepsizes')
    parser.add_argument('--minStepsize', type=int, default=-5, help='x: min stepsize is exp(x)')
    parser.add_argument('--maxStepsize', type=int, default=2, help='x: max stepsize is exp(x)')
    parser.add_argument('--nStepsizeL', type=int, default=1, help='')
    parser.add_argument('--minStepsizeL', type=int, default=0, help='')
    parser.add_argument('--maxStepsizeL', type=int, default=0, help='')
    parser.add_argument('--M', type=int, default=10, help='Number of silos')
    parser.add_argument('--Mavail', type=int, default=10, help='Number of available silos')
    parser.add_argument('--RList', type=list, default=[50], help='List of rounds')
    parser.add_argument('--R_local', type=int, default=50, help='')
    parser.add_argument('--K_constant', type=int, default=24, help='')
    parser.add_argument('--p', type=float, default=0.0, help='p=0.0 for full heterogeneity, p=1.0 for full homogeneity')
    parser.add_argument('--q_pct_of_data', type=float, default=0.2, help='Portion of data')
    parser.add_argument('--Ls', type=list, default=[0.01, 0.1, 1, 10, 100], help='Clipping thresholds')
    parser.add_argument('--qRList', type=list, default=[0], help='')
    parser.add_argument('--epsilons', type=list, default=[0.75, 1, 1.5, 3, 6, 12, 18], help='')
    # 11: Noisy spider, 12: non-private spider, 21: Noisy MB-SGD, 22: non-private MB-SGD, 31: Noisy local SGD, 32: non-private local SGD
    # 41: Noisy SPIDER-Boost, 42: non-private SPIDER-Boost
    parser.add_argument('--models2run', type=list, default=[11, 12, 21, 22, 41, 42], help='')
    args = parser.parse_args()

    for argKey, argValue in args.__dict__.items():
        print(f"{argKey}: {argValue}")

    for args.R in args.RList:
        main(args, args.seed)


