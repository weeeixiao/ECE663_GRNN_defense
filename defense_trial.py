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
from Backbone import *


def my_seed():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)


def grnn_attack_on_certain_epoch(true_g, net, num_classes, img_record):
    # num_classes = 10
    # Gnet = Generator(num_classes, channel=1, shape_img=args.shape_img[0], batchsize=batch_size, g_in=args.g_in).cuda(args.device0)
    Gnet = Generator(num_classes, channel=3, shape_img=args.shape_img[0], batchsize=args.batchsize, g_in=args.g_in).cuda(args.device0)
    Gnet.weight_init(mean=.0, std=.02)
    G_optimizer = torch.optim.RMSprop(Gnet.parameters(), lr=0.0001, momentum=0.99)
    tv_loss = TVLoss()

    G_ran_in = torch.randn(args.batchsize, args.g_in).cuda(args.device0)
    history, history_l = [], []
    for iter in tqdm(range(args.iteration), dynamic_ncols=True):
        Gout, Glabel = Gnet(G_ran_in)
        Gout, Glabel = Gout.cuda(args.device1), Glabel.cuda(args.device1)
        # _, _, grad = client.train(net=copy.deepcopy(net_glob).to(args.device))

        Gpred = net(Gout)
        Gloss = -torch.mean(torch.sum(Glabel * torch.log(torch.softmax(Gpred, 1)), dim=-1))
        G_dy_dx = torch.autograd.grad(Gloss, net.parameters(), create_graph=True, retain_graph=True)

        fake_g = flatten_gradients(G_dy_dx).cuda(args.device1)

        grad_diff_l2 = loss_f('l2', fake_g, true_g, args.device1)
        grad_diff_wd = loss_f('wd', fake_g, true_g, args.device1)
        tvloss = (1e-3 if args.net_name == 'lenet' else 1e-6) * tv_loss(Gout)
        grad_diff = grad_diff_l2 + grad_diff_wd + tvloss

        G_optimizer.zero_grad()
        grad_diff.backward()
        G_optimizer.step()

        if iter % (args.iteration // args.plot_num) == 0:
            history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(args.batchsize)])
            history_l.append([Glabel.argmax(dim=1)[imidx].item() for imidx in range(args.batchsize)])

        # Clear up the space
        torch.cuda.empty_cache()
        del Gloss, G_dy_dx, flatten_fake_g, grad_diff_l2, grad_diff_wd, grad_diff, tvloss

    # -----------------------------------------------------------------------------------------------
    #                                     Visualize the result
    # -----------------------------------------------------------------------------------------------
    for imidx in range(args.batchsize):
        plt.figure(figsize=(12, 8))
        plt.subplot(args.plot_num // 10, 10, 1)
        plt.imshow(tp(img_record[imidx].cpu()))
        for i in range(min(len(history), args.plot_num - 1)):
            plt.subplot(args.plot_num // 10, 10, i + 2)
            plt.imshow(history[i][imidx])
            plt.title(f'l={history_l[i][imidx]}')
            plt.axis('off')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the true and fake figures
        if args.save_img:
            true_path = os.path.join(save_img_path, f'true_data/exp{str(idx_net).zfill(3)}/')
            fake_path = os.path.join(save_img_path, f'fake_data/exp{str(idx_net).zfill(3)}/')
            os.makedirs(true_path, exist_ok=True)
            os.makedirs(fake_path, exist_ok=True)
            tp(gt_data[imidx].cpu()).save(os.path.join(true_path, f'{imidx}_{gt_label[imidx].item()}.png'))
            history[-1][imidx].save(os.path.join(fake_path, f'{imidx}_{Glabel.argmax(dim=1)[imidx].item()}.png'))

        plt.savefig(f'{save_path}/exp:{idx_net:03d}-imidx:{imidx:02d}-tlabel:{gt_label[imidx].item()}-Glabel:{Glabel.argmax(dim=1)[imidx].item()}.png')
        plt.close()

    # Clear up the space
    del Glabel, Gout, flatten_true_g, G_ran_in, net, Gnet
    torch.cuda.empty_cache()
    history.clear()
    history_l.clear()

if __name__ == '__main__':
    my_seed()

    args = args_parser()
    os.environ["CUDA_DEVICE_ORDER"] = args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train = None

    # --------------------------------------------------------------------------------
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    # dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    # args.num_channels = 1
    dataset_train, num_classes = gen_dataset(args.dataset, args.data_path, args.shape_img)
    tp = transforms.Compose([transforms.ToPILImage()])
    # train_loader = iter(torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, generator=torch.Generator(device='cuda'),))
    # --------------------------------------------------------------------------------
    
    # sample users
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users = mnist_noniid(dataset_train, args.num_users)
    img_size = dataset_train[0][0].shape

    # args.model == 'cnn'
    # net_glob = CNNMnist(args=args).to(args.device)
    # net_glob = CNNCifar(args=args).to(args.device)
    # if args.dp_mechanism != 'no_dp':
    #     net_glob = GradSampleModule(net_glob)
    # print(net_glob)
    # net_glob.train()

    net_glob = LeNet(num_classes=num_classes) if args.net_name == 'lenet' else ResNet18(num_classes=num_classes)
    net_glob = net_glob.cuda(args.device1)
    if args.dp_mechanism != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    net_glob.train()

    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # claim clients
    # --------------------------------------------------------------------------------
    # if args.serial:
    #     clients = [LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    # else:
    #     clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i], batchsize=args.batchsize) for i in range(args.num_users)]
    # --------------------------------------------------------------------------------

    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)

    batchsize_record = 0
    net_glob_record = None
    dy_dx_record = None
    img_record = None

    for iter in range(args.epochs):
        # t_start = time.time()
        w_locals, loss_locals, weight_locals = [], [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]
        for idx in tqdm(idxs_users):
            local = clients[idx]
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # XW: temporary implementation
            # --------------------------------------------------------------------------------
            if iter == 0 and dy_dx_record is None:
                w, loss, dy_dx_record, net_glob_record, img_record = local.record_train(net=copy.deepcopy(net_glob).to(args.device))
            # --------------------------------------------------------------------------------

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locals.append(len(dict_users[idx]))

        # update global weights
        print("Enter weight avg phase of epoch {iter}")
        w_glob = FedWeightAvg(w_locals, weight_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

    # XW: current trial, not examined
    # --------------------------------------------------------------------------------
    # dy_dx = torch.autograd.grad(loss_record, net_glob.parameters())
    true_g = flatten_gradients(dy_dx_record)
    net_copy = LeNet(num_classes=num_classes) if args.net_name == 'lenet' else ResNet18(num_classes=num_classes)
    net_copy = net_copy.cuda(args.device1)
    if args.dp_mechanism != 'no_dp':
        net_glob_record = net_glob_record._module
    net_copy.load_state_dict(net_glob_record.state_dict())
    net_copy.eval()
    grnn_attack_on_certain_epoch(true_g, net_copy, num_classes, img_record)
    # --------------------------------------------------------------------------------
