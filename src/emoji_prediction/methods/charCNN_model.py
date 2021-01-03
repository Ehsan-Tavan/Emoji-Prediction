#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
charCnn_model.py is written for charCnn model
https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
"""

from torch import nn

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/03/2021"


class CharCnn(nn.Module):
    """
    In this class we implement charCnn model
    """
    def __init__(self, **kwargs):
        super().__init__()

        # Embedding Layer
        self.embeddings = nn.Embedding(kwargs["vocab_size"], kwargs["embed_size"])
        self.embeddings.weight = nn.Parameter(kwargs["embeddings"], requires_grad=False)

        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=kwargs["embed_size"],
                      out_channels=kwargs["num_channels"],
                      kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )  # (batch_size, num_channels, (seq_len-6)/3)
        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=kwargs["num_channels"],
                      out_channels=kwargs["num_channels"],
                      kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )  # (batch_size, num_channels, (seq_len-6-18)/(3*3))
        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=kwargs["num_channels"],
                      out_channels=kwargs["num_channels"],
                      kernel_size=3),
            nn.ReLU()
        )  # (batch_size, num_channels, (seq_len-6-18-18)/(3*3))
        conv4 = nn.Sequential(
            nn.Conv1d(in_channels=kwargs["num_channels"],
                      out_channels=kwargs["num_channels"],
                      kernel_size=3),
            nn.ReLU()
        )  # (batch_size, num_channels, (seq_len-6-18-18-18)/(3*3))
        conv5 = nn.Sequential(
            nn.Conv1d(in_channels=kwargs["num_channels"],
                      out_channels=kwargs["num_channels"],
                      kernel_size=3),
            nn.ReLU()
        )  # (batch_size, num_channels, (seq_len-6-18-18-18-18)/(3*3))
        conv6 = nn.Sequential(
            nn.Conv1d(in_channels=kwargs["num_channels"],
                      out_channels=kwargs["num_channels"],
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )  # (batch_size, num_channels, (seq_len-6-18-18-18-18-18)/(3*3*3))

        # Length of output after conv6
        conv_output_size = kwargs["num_channels"] * ((kwargs["seq_len"] - 96) // 27)

        linear1 = nn.Sequential(
            nn.Linear(conv_output_size, kwargs["linear_size"]),
            nn.ReLU(),
            nn.Dropout(kwargs["dropout"])
        )
        linear2 = nn.Sequential(
            nn.Linear(kwargs["linear_size"], kwargs["linear_size"]),
            nn.ReLU(),
            nn.Dropout(kwargs["dropout"])
        )
        linear3 = nn.Linear(kwargs["linear_size"], kwargs["output_size"])

        self.convolutional_layers = nn.Sequential(conv1, conv2, conv3, conv4, conv5, conv6)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)

    def forward(self, input_batch):
        # input_batch.size() = [seq_len, batch_size]

        embedded_sent = self.embeddings(input_batch).permute(1, 2, 0)
        # embedded_sent.size() = [batch_size, embed_size, seq_len]

        conv_out = self.convolutional_layers(embedded_sent)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        linear_output = self.linear_layers(conv_out)
        return linear_output
