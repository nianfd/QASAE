import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import causal_convolution_layer

class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self):
        super(TransformerTimeSeries, self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(1, 256)

        self.input_embedding_diff = causal_convolution_layer.context_embedding(1, 256)
        self.fc1 = nn.Linear(512, 256)  ##全连接层
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)  ##全连接层
        #self.fc1 = torch.nn.Linear(130, 256)
        # self.fc1 = torch.nn.Linear(256, 2)
        #
        # self.gau = GAU(
        #     dim=256,
        #     query_key_dim=128,  # query / key dimension
        #     causal=True,  # autoregressive or not
        #     expansion_factor=2,  # hidden dimension = dim * expansion_factor
        #     laplace_attn_fn=True  # new Mega paper claims this is more stable than relu squared as attention function
        # )

    def forward(self, x, x_diff):
        #concatenate observed points and time covariate
        #(B*feature_size*n_time_points)
        #print(x.shape)
        z = x.unsqueeze(1)
        z_diff = x_diff.unsqueeze(1)
        #print(x.shape)
        # print(z.squeeze(1).shape)


        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z)

        z_embedding_diff = self.input_embedding(z_diff)

        fea = torch.cat([z_embedding, z_embedding_diff],dim=1)

        out = F.relu(self.fc1(fea))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def gau_trans(**kwargs):
    model = TransformerTimeSeries(**kwargs)
    return model

