import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule

from Generator.model import Generator
from grnn_utils import *
from tqdm import tqdm


def my_seed():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)


def grnn_attack_on_certain_epoch(true_g, batch_size, client):
    num_classes = 10
    Gnet = Generator(num_classes, channel=1, shape_img=args.shape_img[0], batchsize=batch_size, g_in=args.g_in).cuda(args.device0)
    Gnet.weight_init(mean=.0, std=.02)
    G_optimizer = torch.optim.RMSprop(Gnet.parameters(), lr=0.0001, momentum=0.99)
    tv_loss = TVLoss()

    G_ran_in = torch.randn(args.batchsize, args.g_in).cuda(args.device0)
    for iter in tqdm(range(args.iteration), dynamic_ncols=True):
        Gout, Glabel = Gnet(G_ran_in)
        Gout, Glabel = Gout.cuda(args.device1), Glabel.cuda(args.device1)
        _, loss = client.train(net=copy.deepcopy(net_glob).to(args.device))

        G_dy_dx = torch.autograd.grad(loss, net_glob.parameters(), create_graph=True)
        fake_g = flatten_gradients(G_dy_dx).cuda(args.device1)

        grad_diff_l2 = loss_f('l2', fake_g, true_g, args.device1)
        grad_diff_wd = loss_f('wd', fake_g, true_g, args.device1)
        tvloss = (1e-3 if args.net_name == 'lenet' else 1e-6) * tv_loss(Gout)
        grad_diff = grad_diff_l2 + grad_diff_wd + tvloss

        G_optimizer.zero_grad()
        grad_diff.backward()
        G_optimizer.step()



if __name__ == '__main__':
    my_seed()
    # random.seed(123)
    # np.random.seed(123)
    # torch.manual_seed(123)
    # torch.cuda.manual_seed_all(123)
    # torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # args.dataset == 'mnist'
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    args.num_channels = 1
    # sample users
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users = mnist_noniid(dataset_train, args.num_users)
    img_size = dataset_train[0][0].shape

    # args.model == 'cnn'
    net_glob = CNNMnist(args=args).to(args.device)
    if args.dp_mechanism != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # claim clients
    if args.serial:
        clients = [LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    else:
        clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)

    loss_record = None
    batchsize_record = 0
    idx_record = 0
    for iter in range(args.epochs):
        # t_start = time.time()
        w_locals, loss_locals, weight_locols = [], [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]
        for idx in idxs_users:
            local = clients[idx]
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # XW: temporary implementation
            if iter == args.epochs - 1 and loss_record is None:
                loss_record = loss
                batchsize_record = clients[idx].idxs_sample
                idx_record = idx

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        # update global weights
        w_glob = FedWeightAvg(w_locals, weight_locols)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

    # XW: current trial, not examined
    dy_dx = torch.autograd.grad(loss_record, net_glob.parameters())
    true_g = flatten_gradients(dy_dx)
    grnn_attack_on_certain_epoch(true_g, batchsize_record, clients[idx_record])
