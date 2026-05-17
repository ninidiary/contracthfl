#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Federated Learning Main Script using FedNova optimizer (https://github.com/JYWa/FedNova/tree/master).
"""

import os
import numpy as np
import time
import argparse
import sys
import csv
import copy
import pandas as pd
from models import *
import math
from math import ceil
from random import Random
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from distoptim import FedProx, FedNova


def args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-n', default="test_fednova", type=str, help='experiment name')
    parser.add_argument('--backend', default="nccl", type=str, help='background name')
    parser.add_argument('--model', default="VGG", type=str, help='neural network model')
    parser.add_argument('--NIID', action='store_true', default=True, help='whether the dataset is non-iid or not')
    parser.add_argument('--alpha', default=0.2, type=float, help='control the non-iidness of dataset')
    parser.add_argument('--gmf', default=0, type=float, help='global (server) momentum factor')
    parser.add_argument('--lr', default=0.01, type=float, help='client learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='local (client) momentum factor')
    parser.add_argument('--bs', default=16, type=int, help='batch size on each worker/client')
    parser.add_argument('--rounds', default=10, type=int, help='total communication rounds')
    parser.add_argument('--localE', default=20, type=int, help='number of local epochs')
    parser.add_argument('--meanE', default=10, type=int, help='average number of local epochs')
    parser.add_argument('--print_freq', default=100, type=int, help='print info frequency')
    parser.add_argument('--size', default=8, type=int, help='number of local workers')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--save', '-s', action='store_true', help='whether save the training results')
    parser.add_argument('--p', '-p', action='store_true', help='whether the dataset is partitioned or not')
    parser.add_argument('--pattern', type=str, help='pattern of local steps')
    parser.add_argument('--optimizer', default='fednova', type=str, help='optimizer name')
    parser.add_argument('--initmethod', default='tcp://h0:22000', type=str, help='init method')
    parser.add_argument('--mu', default=1, type=float, help='mu parameter in fedprox')
    parser.add_argument('--savepath', default='./results/', type=str, help='directory to save exp results')
    parser.add_argument('--datapath', default='./data2/', type=str, help='directory to load data')
    args = parser.parse_known_args()[0]
    return args


def update_learning_rate(optimizer, epoch, target_lr):
    """Decay learning rate exponentially at specific epochs."""
    if epoch == int(args.rounds / 2):
        lr = target_lr / 10
        print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch == int(args.rounds * 0.75):
        lr = target_lr / 100
        print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Meter(object):
    """Computes and stores the average, variance, and current value."""

    def __init__(self, init_dict=None, ptag='Time', stateful=False, csv_format=True):
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count) / (self.count - 1)) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'.format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'.format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'.format(dm=self))
            else:
                return str(self.ptag) + str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'.format(dm=self))


class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partitions a dataset into different chunks."""

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False, alpha=0, dataset=None):
        self.data = data
        self.dataset = dataset
        if isNonIID:
            print('Dataset is Non IID!')
            self.partitions, self.ratio = self.__getDirichletData__(data, sizes, seed, alpha)
        else:
            print('Dataset is IID!')
            self.partitions = []
            self.ratio = sizes
            rng = Random()
            rng.seed(seed)
            data_len = len(data)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)

            for frac in sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getDirichletData__(self, data, psizes, seed, alpha):
        n_nets = len(psizes)
        K = 10
        labelList = np.array(data.targets)
        min_size = 0
        N = len(labelList)
        np.random.seed(2020)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)
        return idx_batch, weights


def partition_dataset(rank, size, args):
    """Load and partition the CIFAR-10 dataset."""
    print('==> load train data')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.datapath, train=True, download=True, transform=transform_train)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(trainset, partition_sizes, isNonIID=args.NIID, alpha=args.alpha)
    ratio = partition.ratio
    partition = partition.use(rank)
    train_loader = torch.utils.data.DataLoader(partition, batch_size=args.bs, shuffle=True, pin_memory=True)

    print('==> load test data')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.datapath, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=size)

    return train_loader, test_loader, ratio


def select_model(num_class, args):
    """Select the neural network model."""
    if args.model == 'VGG':
        model = vgg11()
    return model


def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, criterion, optimizer, loader, epoch):
    """Training step for one batch."""
    model.train()
    losses = Meter(ptag='Loss')
    top1 = Meter(ptag='Prec@1')

    for batch_idx, (data, target) in enumerate(loader):
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_acc = comp_accuracy(output, target)
        losses.update(loss.item(), data.size(0))
        top1.update(train_acc[0].item(), data.size(0))

    return model.state_dict()


def average_weights(w):
    """Returns the average of the weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def evaluate(model, test_loader):
    """Evaluate the model on the test set."""
    model.eval()
    top1 = Meter(ptag='Acc')
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            outputs = model(data)
            acc1 = comp_accuracy(outputs, target)
            top1.update(acc1[0].item(), data.size(0))
            batch_loss = nn.CrossEntropyLoss()(outputs, target)
            loss += batch_loss.item()
    return top1.avg, loss


def generate_random_array(size, mean, lower_bound, upper_bound):
    """Generate a random array with a specific mean."""
    random_array = np.random.randint(lower_bound, upper_bound + 1, size)
    current_mean = np.mean(random_array)
    adjustment_factor = mean / current_mean
    adjusted_array = np.clip(random_array * adjustment_factor, lower_bound, upper_bound).astype(int)
    return adjusted_array


def get_Fnw(global_model, train_loader):
    """Compute the mean gradient (Fw) for the global model."""
    global_model.train()
    total_grad = None
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        images.requires_grad_(True)

        global_model.zero_grad()
        target_pred = global_model(images)
        loss = nn.CrossEntropyLoss()(target_pred, labels)
        gradients = torch.autograd.grad(loss, global_model.parameters(), create_graph=False)

        batch_grad = torch.cat([g.flatten() for g in gradients])
        if total_grad is None:
            total_grad = batch_grad
        else:
            total_grad += batch_grad
        total_samples += 1

    mean_grad = total_grad / total_samples
    return mean_grad.unsqueeze(0)


def get_betan(rank):
    """Calculate the beta value for the client."""
    # Note: This function relies on global variables 'means' and 'Fw' defined in the main loop.
    Fnw = means[rank]
    Fnw_norm = Fnw.norm(dim=1, p=2)
    Fw_norm = Fw.norm(dim=1, p=2)
    beta = Fnw_norm / Fw_norm - 1
    beta = beta.detach().cpu().numpy()
    return beta[0]


def main():
    """Main training loop."""
    args = args_parser()
    if not os.path.exists('./log/' + args.exp_name):
        os.makedirs('./log/' + args.exp_name)

    size = args.size
    device = 'cuda:0'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    global_model = select_model(10, args).to(device)
    global_weights = global_model.state_dict()

    algorithms = {
        'fedavg': FedProx,
        'fedprox': FedProx,
        'fednova': FedNova,
    }
    selected_opt = algorithms[args.optimizer]

    epochs_array = generate_random_array(size, mean=args.meanE, lower_bound=1, upper_bound=args.localE)
    print(epochs_array)

    test_accs, test_losses = [], []
    betas = np.zeros([args.rounds, args.size])
    results = pd.DataFrame()

    for rnd in range(args.rounds):
        start = time.time()
        global_model.train()
        local_weights = []
        means = []

        for rank in range(args.size):
            train_loader, test_loader, DataRatios = partition_dataset(rank, size, args)
            local_epochs = epochs_array[rank]
            print('global round:{:d}, client id={:d}, localE={:d}'.format(rnd, rank, local_epochs))

            local_model = copy.deepcopy(global_model)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = selected_opt(local_model.parameters(),
                                     lr=args.lr,
                                     gmf=args.gmf,
                                     mu=args.mu,
                                     ratio=DataRatios[rank],
                                     momentum=args.momentum,
                                     nesterov=False,
                                     weight_decay=1e-4)

            update_learning_rate(optimizer, rnd, args.lr)

            for t in range(local_epochs):
                w = train(local_model, criterion, optimizer, train_loader, t)
                local_weights.append(copy.deepcopy(w))

            mean_rank = get_Fnw(global_model, train_loader)
            means.append(mean_rank)

        Fw = torch.mean(torch.stack(means), dim=0)
        Fw_norm = Fw.norm(dim=1, p=2)
        Fw_norm_values = [tensor.item() for tensor in Fw_norm]

        for rank in range(args.size):
            beta = get_betan(rank)
            betas[rnd, rank] = beta

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        test_acc, test_loss = evaluate(global_model, test_loader)
        print('global round:{:d}, test_acc={:.2f}, test_loss={:.4f}'.format(rnd, test_acc, test_loss))
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        end = time.time()
        cost_time = end - start
        print(f'Round {rnd} takes {cost_time} s.')

        file_name = f'./exp_{args.optimizer}_{args.model}_{args.NIID}_alpha{args.alpha}_lr{args.lr}_bs{args.bs}_epoch{args.meanE}_{args.localE}_size{args.size}'
        torch.save(global_model.state_dict(), f'{file_name}_model.pth')
        print(f'saved round {rnd} model gradient!')

        results['round'] = list(range(len(test_accs)))
        results['acc'] = test_accs
        results['loss'] = test_losses
        results['Fw_norm'] = Fw_norm_values
        results.to_csv(file_name + '.csv', index=False)
        pd.DataFrame(betas).to_csv(file_name + '_beta.csv', index=False)


if __name__ == "__main__":
    main()
