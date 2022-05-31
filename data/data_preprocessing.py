#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

"""
from data.data_functions import normalize, normalize2, coo2tensor, compute_projection_matrix

import torch
import pickle
import pandas as pd 
import numpy as np
from collections import defaultdict 


class CollaborationComplex(torch.utils.data.Dataset):
    def __init__(self, pct_miss, order, id_rlz, eps=0.9, kappa=5, starting_node=150250, data_path=r"data/collaboration_complex",):
        """
        pct_miss: the missing percentage 10,20,30,...,,60
        order: the simplices order 0,1,2,...,5
        id_rlz: the id of the realizations, 0,1,...,9 (10 realizations in total)
        eps, kappa: the epsilon and order in computing the projection matrix in SAT architecture 
        device: "cpu" or "gpu"
        """
        assert order >= 0
        assert pct_miss in range(10, 60, 10)
        self.incidences = np.load('{}/{}_boundaries.npy'.format(data_path, starting_node), allow_pickle=True)

        # workaround order == len(self.incidences)
        # is not taken into account at the moment since is higher than
        # the maximum number used in the experiments.
 
        L_u = np.load("{}/{}_laplacians_up.npy".format(data_path, starting_node), allow_pickle=True)[order]
        L_l = np.load("{}/{}_laplacians_down.npy".format(data_path, starting_node), allow_pickle=True)[order]
        L = np.load("{}/{}_laplacians.npy".format(data_path, starting_node), allow_pickle=True)[order]

        # convert the sparse npy matrix into dense tensors 
        L_l = coo2tensor(normalize2(L, L_l ,half_interval=True)).to_dense()
        L_u = coo2tensor(normalize2(L, L_u ,half_interval=True)).to_dense()
        L_hodge = coo2tensor(normalize2(L, L ,half_interval=True)).to_dense()

        # self.L = (Ldo, Lup, compute_projection_matrix(L1, eps=eps, kappa=kappa))
        # the projection matrix in SAT
        projection_matrix = compute_projection_matrix(L_hodge, eps=eps, kappa=kappa)
        
        # input signals where the missing values are replaced by the median values 
        observed_signal = np.load('{}/{}_percentage_{}_input_damaged_{}.npy'.format(
            data_path, starting_node, pct_miss, id_rlz), allow_pickle=True)
        # covnert to tensors     
        observed_signal = [torch.tensor(
            list(signal.values()), dtype=torch.float) for signal in observed_signal]

        # true signals 
        target_signal = np.load(
            '{}/{}_cochains.npy'.format(data_path, starting_node), allow_pickle=True)
        # convert to tensors     
        target_signal = [torch.tensor(
            list(signal.values()), dtype=torch.float) for signal in target_signal]

        # masks
        masks = np.load('{}/{}_percentage_{}_known_values_{}.npy'.format(data_path, starting_node, pct_miss, id_rlz), allow_pickle=True)  # positive mask= indices that we keep ##1 mask #entries 0 degree 
        # convert to tensors 
        masks = [torch.tensor(list(mask.values()), dtype=torch.long) for mask in masks]
        print(len(masks))

        # only take the corresponding order, the order of interest 
        self.X = observed_signal[order] 
        print(self.X)
        
        self.y = target_signal[order]
        print(self.y)
        self.n = len(self.X)
        self.mask = masks[order]
        self.L = L_hodge
        self.L_l = L_l
        self.L_u = L_u
        self.proj = projection_matrix

    def __getitem__(self, index):
        # if self.mask is returned too, then we evaluate the loss over the known values, and we can also evaluate the accuracy only on the predicted ones; otherwise, both loss and accuracy are evaluated on the whole list
        return self.X, self.y, self.mask

    def __len__(self):
        # Returns length
        return 1