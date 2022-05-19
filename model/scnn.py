"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

code for building, training and testing the SCNN model

paper: https://arxiv.org/abs/2110.02585
"""

import os 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 
import scipy.sparse as sp

class scnn_conv(nn.Module):
    def __init__(self, F_in, F_out, K1, K2, laplacian_l, laplacian_u, alpha_leaky_relu):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        K1, K2: the filter lower and upper orders, on the lower and upper laplacians respectively
        alpha_leaky_relu: the negative slop of the leaky relu function 
        """
        super(scnn_conv, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.K1 = K1
        self.K2 = K2 
        self.W = nn.parameter.Parameter(torch.empty(size=(self.K, self.F_in, self.F_out))) # define and initialize the filter weights, which is of dimension K x F_in x F_out 
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)
        self.L = laplacian 
        self.reset_parameters()
        print("created SNN layers")

    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)

    def forward(self,x):
        """
        define the simplicial convolution in the SNN architecture (i.e., the simplicial filtering operation)
        x: input features of dimension M x F_in (num_edges/simplices x num_input features)
        """
        
        L = self.L 
        L_k = torch.eye(L.size(dim=0)).requires_grad_(False) # the identity matrix
        y_k = L_k @ torch.clone(x @ self.W[0]) # this is the 0th term
        for k in range(1,self.K):
            L_k = L_k @ L 
            y_k += L_k @ x @ self.W[k]

        return y_k 