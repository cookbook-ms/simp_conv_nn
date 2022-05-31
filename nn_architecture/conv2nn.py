#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

build the SNN model by stacking the snn_conv layers with nonlinearity

paper: https://arxiv.org/abs/2010.03633
"""

import sys
# from turtle import forward
# from sympy import hyper
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torchmetrics
import torch.nn as nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.snn_conv import snn_conv
from model.scnn_conv import scnn_conv
from model.san_conv import san_conv
from model.psnn_conv import psnn_conv 

spmm = torch.sparse.mm

class conv2nn(pl.LightningModule):
    def __init__(self, F_in, F_intermediate, F_out, K, K1, K2, laplacian, laplacian_l, laplacian_u, projection_matrix, sigma, device, model):
        """
        Parameters
        ----------
        - F_in: number of the input features : 1
        - F_intermediate: number of intermediate features per layer e.g., [2,5,5] -- 2 outputs in the 2nd layer, 5 outputs in the 3rd and 4th layer, but not including the last layer, which has again 1 output in general 
        - F_out: number of the output features: generally 1

        - K: filter order when using only one shift operator; when K1, K2 are applied, set K as none or 0
        - K1: filter order of the lower shift operator
        - K2: filter order of the upper shift operator

        - laplacian: the hodge laplacian of the corresponding order 
        - laplacian_l: the lower laplacian of the corresponding order 
        - laplacian_u: the upper laplacian of the corresponding order 
        - projection_matrix: harmonic space projection matrix

        - sigma: the chosen nonlinearity, e.g., nn.LeakyReLU()
        - alpha_leaky_relu: the negative slope of the leakyrelu, if applied

        - device: "cpu" or "gpu"

        - architecture: choose the architecture - "snn", "scnn", "san", "psnn" 
        """
        super(conv2nn, self).__init__()
        self.num_features = [F_in] + [F_intermediate[l] for l in range(len(F_intermediate))] + [F_out] # number of features vector e.g., [1 5 5 5 1]
        self.num_layers = len(self.num_features) 

        self.K = K
        self.K1 = K1
        self.K2 = K2 
        print(K1,K2)
        self.L = laplacian
        self.L_l = laplacian_l
        self.L_u = laplacian_u 
        self.proj = projection_matrix

        self.sigma = sigma 

        nn_layer = []
        # define the NN layer operations for each model
        if model == 'snn':
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"K":self.K,"laplacian":self.L}
                nn_layer.extend([snn_conv(**hyperparameters).to(device), self.sigma])

        elif model == 'scnn': 
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"K1":self.K1, "K2":self.K2,"laplacian_l":self.L_l,"laplacian_u":self.L_u}
                nn_layer.extend([scnn_conv(**hyperparameters).to(device), self.sigma])

        elif model == 'psnn':
            for l in range(self.num_layers-1): 
                hyperparameters = {
                    "F_in":self.num_features[l], "F_out":self.num_features[l+1],"laplacian_l":self.L_l,"laplacian_u":self.L_u
                }
                nn_layer.extend([psnn_conv(**hyperparameters).to(device), self.sigma])

        elif model == 'san':
            for l in range(self.num_layers-1):
                hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"K1":self.K1,"K2":self.K2,"laplacian_l":self.L_l,"laplacian_u":self.L_u,"projection_matrix":self.proj}
                nn_layer.extend([san_conv(**hyperparameters).to(device), self.sigma])

        else: 
            raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)
        self.loss_fn = nn.L1Loss(reduction='mean')
        #self.train_acc = torchmetrics.Accuracy()
        #self.val_acc = torchmetrics.Accuracy()
        self.max_acc = 0.0

    def forward(self,x):
        return self.simplicial_nn(x).view(-1,1).T

    def training_step(self, batch, batch_idx): 
        if len(batch) == 3:
            x, y, mask = batch 
            print("loss and acc evaluated on the known and unknown")
        else:
            x, y = batch
            mask = range(len(y))
            print("loss and acc evaluated on the whole")
        
        print(len(mask)/len(y))
        print(x)
        y_hat = self(x).squeeze(0)
        print(len(y_hat))
        loss_in = self.loss_fn(x[mask],y[mask])
        loss = self.loss_fn(y_hat[mask],y[mask])
        #self.train_acc(y_hat,y)
        #self.val_acc(y_hat,y)
        t1 = mask.numpy() 
        #print(len(t1))
        t2 = np.array(range(len(y)))
        #print(t2)
        mask_pred = torch.tensor(np.setdiff1d(t2,t1))
        self.acc = ((y[mask_pred].float()-y_hat[mask_pred]).abs() <= (0.05*y[mask_pred]).abs()).sum() / len(y[mask_pred])
        print(mask)
        print((y[mask]-x[mask]).size())
        print(mask_pred)
        print((x[mask_pred].float()))
        print(y[(range(len(y)))])
        
        self.acc_in = ((y.float()-x).abs() <= (0.05*y).abs()).sum() / len(y)
        
        self.max_acc = max(self.acc, self.max_acc) 
        self.log('input_acc:', self.acc_in, on_step=False, on_epoch=True, prog_bar=True)
        self.log('training_acc:', self.acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('input_loss:', loss_in.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('training_loss:', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss 
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = range(len(y))
        #y = y.unsqueeze(0)
        y_hat = self(x).squeeze(0)

        loss = self.loss_fn(y_hat[mask], y[mask])
        return loss 
        
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def training_epoch_end(self, outs):
        pass
    
    def validation_epoch_end(self, outs):
        pass 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=0.0)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.77,
                                      patience=100,
                                      min_lr=7e-5,
                                      verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'training_loss:'}