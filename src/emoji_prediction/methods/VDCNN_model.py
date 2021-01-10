#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
VDCNN_model.py is a module for VDCNN model
https://github.com/ArdalanM/nlp-benchmarks
"""

import torch
from torch import nn

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/10/2021"


class BasicConvResBlock(nn.Module):

    def __init__(self, **kwargs):
        super(BasicConvResBlock, self).__init__()

        self.downsample = kwargs["downsample"]
        self.shortcut = kwargs["shortcut"]

        self.conv1 = nn.Conv1d(in_channels=kwargs["input_dim"],
                               out_channels=kwargs["n_filters"],
                               kernel_size=kwargs["kernel_size"],
                               padding=kwargs["padding"],
                               stride=kwargs["stride"])
        self.bn1 = nn.BatchNorm1d(kwargs["n_filters"])
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=kwargs["n_filters"],
                               out_channels=kwargs["n_filters"],
                               kernel_size=kwargs["kernel_size"],
                               padding=kwargs["padding"],
                               stride=kwargs["stride"])
        self.bn2 = nn.BatchNorm1d(kwargs["n_filters"])

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out


class VDCNN(nn.Module):
    def __init__(self, **kwargs):
        super(VDCNN, self).__init__()
        layers = []
        fc_layers = []

        depth = kwargs["depth"]
        shortcut = kwargs["shortcut"]

        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])

        layers.append(nn.Conv1d(in_channels=kwargs["embedding_dim"],
                                out_channels=64,
                                kernel_size=3,
                                padding=1))

        if depth == 9:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 8, 8, 5, 3

        layers.append(BasicConvResBlock(input_dim=64,
                                        n_filters=64,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        downsample=None,
                                        shortcut=shortcut))

        for _ in range(n_conv_block_64 - 1):
            layers.append(BasicConvResBlock(input_dim=64,
                                            n_filters=64,
                                            kernel_size=3,
                                            padding=1,
                                            stride=1,
                                            downsample=None,
                                            shortcut=shortcut))

        layers.append(nn.MaxPool1d(kernel_size=3,
                                   stride=2,
                                   padding=1))  # l = initial length / 2

        ds = nn.Sequential(nn.Conv1d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=1,
                                     stride=1,
                                     bias=False),
                           nn.BatchNorm1d(128))

        layers.append(BasicConvResBlock(input_dim=64,
                                        n_filters=128,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        downsample=ds,
                                        shortcut=shortcut))

        for _ in range(n_conv_block_128-1):
            layers.append(BasicConvResBlock(input_dim=128,
                                            n_filters=128,
                                            kernel_size=3,
                                            padding=1,
                                            stride=1,
                                            downsample=None,
                                            shortcut=shortcut))

        layers.append(nn.MaxPool1d(kernel_size=3,
                                   stride=2,
                                   padding=1))  # l = initial length / 4

        ds = nn.Sequential(nn.Conv1d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1,
                                     bias=False),
                           nn.BatchNorm1d(256))

        layers.append(BasicConvResBlock(input_dim=128,
                                        n_filters=256,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        downsample=ds,
                                        shortcut=shortcut))

        for _ in range(n_conv_block_256 - 1):
            layers.append(BasicConvResBlock(input_dim=256,
                                            n_filters=256,
                                            kernel_size=3,
                                            padding=1,
                                            stride=1,
                                            downsample=None,
                                            shortcut=shortcut))

        layers.append(nn.MaxPool1d(kernel_size=3,
                                   stride=2,
                                   padding=1))

        ds = nn.Sequential(nn.Conv1d(in_channels=256,
                                     out_channels=512,
                                     kernel_size=1,
                                     stride=1,
                                     bias=False),
                           nn.BatchNorm1d(512))

        layers.append(BasicConvResBlock(input_dim=256,
                                        n_filters=512,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        downsample=ds,
                                        shortcut=shortcut))

        for _ in range(n_conv_block_512 - 1):
            layers.append(BasicConvResBlock(input_dim=512,
                                            n_filters=512,
                                            kernel_size=3,
                                            padding=1,
                                            stride=1,
                                            downsample=None,
                                            shortcut=shortcut))

        layers.append(nn.AdaptiveMaxPool1d(8))

        fc_layers.extend([nn.Linear(in_features=8 * 512,
                                    out_features=kwargs["n_fc_neurons"]),
                          nn.ReLU()])

        fc_layers.extend([nn.Linear(in_features=kwargs["n_fc_neurons"],
                                    out_features=kwargs["n_fc_neurons"]),
                          nn.ReLU()])

        fc_layers.extend([nn.Linear(in_features=kwargs["n_fc_neurons"],
                                    out_features=kwargs["n_classes"])])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, sen_len]

        embedded = self.embeddings(input_batch)
        # embedded.size() = [batch_size, sen_len, embedding_dim]

        embedded = embedded.transpose(1, 2)
        # embedded.size() = [batch_size, embedding_dim, sen_len]

        out = self.layers(embedded)
        # out.size() = [batch_size, 512, 8]

        out = out.view(out.size(0), -1)
        # out.size() = [batch_size, 512*8]

        out = self.fc_layers(out)
        # out.size() = [batch_size, n_class]

        return out


model = VDCNN(vocab_size=141, embedding_dim=16, n_classes=15, depth=9,
              pad_idx=0, n_fc_neurons=2048, shortcut=False)

text = torch.rand((32, 200))

model.forward(input_batch=text.long())
