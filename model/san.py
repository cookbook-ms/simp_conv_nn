#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

build the SAN architecture convolution

paper: https://arxiv.org/pdf/2203.07485.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class san_conv(nn.Module):

    def __init__(self, F_in, F_out, laplacian_l, laplacian_u, projection_matrix, K1, K2, p_dropout, alpha_leaky_relu):
        """
        F_in: Numer of features of the input signal
        F_out: Numer of features of the output *component*
        projection_matrix: the harmonic projector
        """
        super(san_conv, self).__init__()

        self.K1 = K1
        self.K2 = K2 
        self.F_in = F_in
        self.F_out = F_out
        self.W1= nn.Parameter(torch.empty(size=(self.K1, F_in, F_out)))
        self.W2 = nn.Parameter(torch.empty(size=(self.K2, F_in, F_out)))
        self.W0 = nn.Parameter(torch.empty(size=(F_in, F_out)))

        self.att1 = nn.Parameter(torch.empty(size=(2*F_out*self.K1, 1)))
        self.att2 = nn.Parameter(torch.empty(size=(2*F_out*self.K2, 1)))


        self.dropout = p_dropout  # 0.0#0.6
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)

        self.Ll = laplacian_l
        self.Lu = laplacian_u 
        self.P = projection_matrix

        self.reset_parameters()
        print("Created SAN Layer")

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.att1.data, gain=gain)
        nn.init.xavier_uniform_(self.att2.data, gain=gain)

    def forward(self, x):
        Ld = self.Ll 
        Lu = self.Lu 
        P = self.P


        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x1 = torch.cat([x @ self.W1[k] for k in range(self.K1)], dim=1)
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x2 = torch.cat([x @ self.W2[k] for k in range(self.K2)], dim=1)
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x0 = P @ x @ self.W0

        # Broadcast add
        E1 = self.leakyrelu((x1 @ self.att1[:self.F_out*self.K1, :]) + (
            x1 @ self.att1[self.F_out*self.K1:, :]).T)  # (Ex1) + (1xE) -> (ExE)
        E2 = self.leakyrelu((x2 @ self.att2[:self.F_out*self.K2, :]) + (
            x2 @ self.att2[self.F_out*self.K2:, :]).T)  # (Ex1) + (1xE) -> (ExE)

        
        zero_vec = -9e15*torch.ones_like(E1)
        E1 = torch.where(Ld != 0, E1, zero_vec)
        E2 = torch.where(Lu != 0, E2, zero_vec)

        # Broadcast add
        alpha1 = F.dropout(F.softmax(
            E1, dim=1), self.dropout, training=self.training) # (ExE) -> (ExE)
        alpha2 = F.dropout(F.softmax(
            E2, dim=1), self.dropout, training=self.training) # (ExE) -> (ExE)


        alpha1_k =  torch.clone(alpha1)
        alpha2_k = torch.clone(alpha2)

        z1 = alpha1_k @ torch.clone(x  @ self.W1[0])
        for k in range(1, self.K1):
            alpha1_k = alpha1_k @ Ld # alpha_irr
            z1 += alpha1_k  @  x  @ self.W1[k]

        z2 = alpha2_k @ torch.clone(x  @ self.W2[0])
        for k in range(1, self.K2):
            alpha2_k = alpha2_k @ Lu #alpha_sol
            # (ExE) x (ExF_out) -> (ExF_out)
            z2 += alpha2_k  @  x  @ self.W2[k]

        out = (z1 + z2 + x0)
        return out