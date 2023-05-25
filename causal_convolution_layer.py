#!/usr/bin/env python
# coding: utf-8



import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256):
        super(context_embedding,self).__init__()
        self.causal_convolution1 = CausalConv1d(in_channels,256,kernel_size=5, stride=3)
        self.causal_convolution2 = CausalConv1d(256, 256, kernel_size=5, stride= 3)
        #self.causal_convolution3 = CausalConv1d(256, 256, kernel_size=5, stride=3)
        nn.init.kaiming_normal_(self.causal_convolution1.weight)
        nn.init.kaiming_normal_(self.causal_convolution2.weight)
        #nn.init.kaiming_normal_(self.causal_convolution3.weight)
        #self.m = torch.nn.BatchNorm1d(256)

        # Knum = args.kernel_num  ## 每种卷积核的数量
        # Ks = args.kernel_sizes  ## 卷积核list，形如[2,3,4]
        Dim = 2601
        Knum = 256  ## 每种卷积核的数量
        Ks = [2,3,4,5,6,7,8,9]  ## 卷积核list，形如[2,3,4]


        self.convs = nn.ModuleList([nn.Conv2d(1, Knum, (K, Dim)) for K in Ks])  ## 卷积层

        self.fc = nn.Linear(len(Ks) * Knum, 256)  ##全连接层

    def forward(self,x):
        #print(x)
        x = F.leaky_relu(self.causal_convolution1(x))
        x = F.leaky_relu(self.causal_convolution2(x))
        #x = F.relu(self.causal_convolution3(x))
        #x = self.causal_convolution2(x)
        # print("----------------------------")
        #print(x.shape)
        #x = self.m(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.leaky_relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat(x, 1)  # (N,Knum*len(Ks))

        #x = self.dropout(x)
        logit = self.fc(x)
        return logit
        #return F.relu(x)

