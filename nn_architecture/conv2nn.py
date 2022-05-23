#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

build the SNN model by stacking the snn_conv layers with nonlinearity

paper: https://arxiv.org/abs/2010.03633
"""

import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torchmetrics
import torch.nn as nn

import pytorch_lightning as pl
from model.snn_conv import snn_conv

spmm = torch.sparse.mm

class conv2nn(pl.LightningModule):
    def __init__(self, F_in, F_out, K, K1, K2, laplacian, laplacian_l, laplacian_u, alpha_leaky_relu, device, architecture):
        """
        Parameters
        ----------
        - F_in: number of input features  per layer
        - F_out: number of output features per layer
        - K: filter order when using only one shift operator; when K1, K2 are applied, set K as none or 0
        - K1: filter order of the lower shift operator
        - K2: filter order of the upper shift operator 
        - laplacian: the hodge laplacian of the corresponding order 
        - laplacian_l: the lower laplacian of the corresponding order 
        - laplacian_u: the upper laplacian of the corresponding order 
        - alpha_leaky_relu: the negative slope of the leakyrelu, if applied
        - device: "cpu" or "gpu"
        - architecture: choose the architecture - "snn", "scnn", "san", "psnn" 
        """
        super(conv2nn, self).__init__()

