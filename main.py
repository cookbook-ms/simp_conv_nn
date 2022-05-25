#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

main file to implement different neural network architectures to train and test on different dataset for different tasks
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import argparse

from nn_architecture.conv2nn import conv2nn 
from data.data_preprocessing import CollaborationComplex  
from data import * 

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

# collaboration complex 
cc = CollaborationComplex(pct_miss=50, order=5, id_rlz=2)

# print(cc.n)
# print(cc.X)
# print(cc.L.shape)
# print(cc.L_l)
# print(cc.L_u)
# print(cc.proj.shape)
simplicial_nn = conv2nn(F_in=1, F_intermediate=[16,16], F_out=1, K=5, K1=2, K2=2, laplacian=cc.L, laplacian_l=cc.L_l, laplacian_u=cc.L_u, projection_matrix=cc.proj, sigma=nn.ReLU(), device=device, model='psnn')

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--max_epochs", help="Maximum number of epochs", type=int, default=10)
parser.add_argument("-att", "--attention", help="Abilitate attention mechanism", type=str, default="T")
parser.add_argument("-f", "--features", help="number of per layer features i.e. [5,5]", type=str, default="[5,5]")
# parser.add_argument("-d", "--dense", help="number of per  dense layer features i.e. [5,5]", type=str,  default="[]")
parser.add_argument("-lr", "--learning_rate", help="learning rate i.e. 0.001", type=float, default=0.001)
parser.add_argument("-wd", "--weight_decay", help="l2 regularization term i.e. 0.001", type=float, default=0.0)
parser.add_argument("-eps", "--eps_proj", help="epsilon value for computing projetion matrix i.e. 0.9", type=float, default=0.0)
parser.add_argument("-Kp", "--k_proj", help="K value for computing projetion matrix i.e. 5", type=int, default=0)
parser.add_argument("-k", "--kappa", help="kappa value for diffusion i.e. 5", type=int, default=5)
parser.add_argument("-do", "--dropout", help="probability of dropout i.e. 0.6", type=float, default=0.0)
parser.add_argument("-a", "--activation", help="activation function all lowercase", type=str, default='leaky_relu')
parser.add_argument("-ns", "--negative_slope", help="negative slope leaky relu", type=float, default=0.01)
parser.add_argument("-pm", "--pct_miss", help="pct of missing values (complex dataset)", type=int, default=10)
parser.add_argument("-o", "--order", help="order of the simplex to load (complex dataset)", type=int, default=0)
parser.add_argument("-en", "--exp_num", help="experimental setup to load (complex dataset)", type=int, default=0)
parser.add_argument("-s", "--seed", help="random seed", type=int, default=0)
parser.add_argument("-id", "--pci_id", help="id bus seed", type=str, default="0")
args = parser.parse_args()


train_loader = torch.utils.data.DataLoader(
        cc, batch_size=None, batch_sampler=None, shuffle=True, num_workers=0)

string = "Test_citation"
logger = pl.loggers.TensorBoardLogger(name=string, save_dir='results')


pl.seed_everything(args.seed)

trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger,
                     gpus=0, auto_select_gpus=False)

trainer.fit(simplicial_nn, train_loader)


print("\n\n######")
print("######")
print("Max Accuracy:", simplicial_nn.max_acc.item())
print("######")
print("######\n\n")