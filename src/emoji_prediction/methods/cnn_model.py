#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
cnn_model.py is written for cnn model
"""

import torch
from torch import nn
import torch.nn.functional as F

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "11/14/2020"


class CNN(nn.Module):
    """
    In this class we implement cnn model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])
        self.embeddings.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=kwargs["n_filters"],
                          kernel_size=(fs, kwargs["embedding_dim"])),
                # nn.BatchNorm2d(kwargs["n_filters"]),
                nn.ReLU(),
                nn.Dropout(kwargs["middle_dropout"])
            )
            for fs in kwargs["filter_sizes"]
        ])

        fc_input_dim = kwargs["n_filters"] * len(kwargs["filter_sizes"])

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=fc_input_dim,
                      out_features=256),
            nn.ReLU(),
            nn.Dropout(kwargs["end_dropout"]),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(kwargs["end_dropout"]),
            nn.Linear(in_features=128, out_features=kwargs["output_size"])
        )

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])

    def forward(self, input_batch):
        # input_batch.size() = (batch_size, sent_len)

        embedded = self.start_dropout(self.embeddings(input_batch))
        # embedded.size() = (batch_size, sent_len, emb_dim)

        embedded = embedded.unsqueeze(1)
        # embedded.size() = (batch size, 1, sent len, emb dim)

        conved = [conv(embedded).squeeze(3) for conv in self.convs]
        # conved_n.size() = (batch size, n_filters, sent len - filter_sizes[n] + 1)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n.size() = (batch size, n_filters)

        cat_cnn = torch.cat(pooled, dim=1)
        # cat_cnn.size() = (batch size, n_filters * len(filter_sizes))

        return self.fully_connected_layers(cat_cnn)
