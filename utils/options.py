#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--dp_mechanism', type=str, default='Gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=20,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='differential privacy clip')
    parser.add_argument('--dp_sample', type=float, default=1, help='sample rate for moment account')

    parser.add_argument('--serial', action='store_true', help='partial serial running to save the gpu memory')
    parser.add_argument('--serial_bs', type=int, default=128, help='partial serial running batch size')

    # from grnn
    parser.add_argument("--cuda_device_order", type=str, default="PCI_BUS_ID", help="Order of CUDA device.")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="CUDA visible devices.")
    parser.add_argument("--device0", type=int, default=0, help="Device for GRNN training.")
    parser.add_argument("--device1", type=int, default=0, help="Device for local training.")

    parser.add_argument("--batchsize", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--save_img", type=bool, default=True, help="Whether to save generated images.")
    parser.add_argument("--iteration", type=int, default=1000, help="Number of optimization steps on GRNN.")
    parser.add_argument("--num_exp", type=int, default=10, help="Experiment number.")
    parser.add_argument("--g_in", type=int, default=128, help="Dimension of GRNN input.")
    parser.add_argument("--plot_num", type=int, default=30, help="Number of plots to generate.")
    parser.add_argument("--net_name", type=str, default="lenet", choices=['lenet', 'res18'], help="Global model name.")

    parser.add_argument("--shape_img", type=tuple, default=(32, 32), help="Shape of the images.")
    parser.add_argument("--root_path", type=str, default='./', help="Root path for the project.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Path for the data.")

    args = parser.parse_args()
    return args
