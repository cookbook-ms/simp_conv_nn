#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

build the PSNN architecture convolution

paper: https://arxiv.org/pdf/2102.10058.pdf
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class psnn_conv(nn.Module):
    def __init__(self, F_in, F_out, laplacian_l, laplacian_u):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        """
        super(psnn_conv, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out))) 
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out))) 

        self.Ll = laplacian_l
        self.Lu = laplacian_u

        self.reset_parameters()
        print("created PSNN layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)

    def forward(self,x):
        """
        define the simplicial convolution in the PSNN architecture (i.e., the subspace-varying simplicial filtering operation but with order 1 on the lower and upper laplacians)
        x: input features of dimension M x F_in (num_edges/simplices x num_input features)
        """
        
        Ll = self.Ll 
        Lu = self.Lu 
        dim_simp = Ll.size(dim=0)
        I = torch.eye(dim_simp).requires_grad_(False) # the identity matrix
        y_0 = I @ torch.clone(x @ self.W0) # this is the 0th term

        y_1 = Ll @ x @ self.W1

        y_2 = Lu @ x @ self.W2 

        return y_0 + y_1 + y_2 