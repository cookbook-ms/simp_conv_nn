#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Maosheng Yang, TU Delft (m.yang-2@tudelft.nl)

main file to implement different neural network architectures to train and test on different dataset for different tasks
"""

import pytorch_lightning as pl
import torch
from nn_architecture import conv2nn 
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

