#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

build the SCNN convolution

paper: https://arxiv.org/abs/2110.02585
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class scnn_conv(nn.Module):
    def __init__(self, F_in, F_out, K1, K2, laplacian_l, laplacian_u):
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
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.K1, self.F_in, self.F_out))) 
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.K2, self.F_in, self.F_out))) 

        self.Ll = laplacian_l
        self.Lu = laplacian_u

        self.reset_parameters()
        print("created SCNN layers")

    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)

    def forward(self,x):
        """
        define the simplicial convolution in the SCNN architecture (i.e., the subspace-varying simplicial filtering operation)
        x: input features of dimension M x F_in (num_edges/simplices x num_input features)
        """
        print(x.size())
        x = torch.reshape(x,(x.size(dim=0),self.F_in))
        print(x.size())
        Ll = self.Ll 
        Lu = self.Lu 
        dim_simp = Ll.size(dim=0)
        I = torch.eye(dim_simp).requires_grad_(False) # the identity matrix
        y_0 = I @ x @ self.W0 # this is the 0th term

        #y_1 = torch.empty(size=(dim_simp, self.F_out))
        
        for k in range(0,self.K1):
            y_0 += Ll @ x @ self.W1[k]
            Ll = Ll @ Ll

        #y_2 = torch.empty(size=(dim_simp, self.F_out))
        
        for k in range(0,self.K2):
            y_0 += Lu @ x @ self.W2[k]
            Lu = Lu @ Lu

        return y_0 #+ y_1 + y_2 