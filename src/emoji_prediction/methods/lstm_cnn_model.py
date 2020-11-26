#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
lstm_cnn_model.py is written for LstmCnn model
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
__date__ = "11/26/2020"


class LstmCnn(nn.Module):
    """
    In this class we implement LstmCnn model
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])
        self.embeddings.weight.requires_grad = True

        self.lstm_1 = nn.LSTM(input_size=kwargs["embedding_dim"],
                              hidden_size=kwargs["lstm_hidden_dim"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.lstm_2 = nn.LSTM(input_size=2 * kwargs["lstm_hidden_dim"],
                              hidden_size=kwargs["lstm_hidden_dim"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=kwargs["n_filters"],
                          kernel_size=(fs, kwargs["embedding_dim"] +
                                       2*2*kwargs["lstm_hidden_dim"])),
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
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=128, out_features=kwargs["output_size"])
        )
        self.dropout = {"start_dropout": nn.Dropout(kwargs["start_dropout"]),
                        "middle_dropout": nn.Dropout(kwargs["middle_dropout"])}

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, sent_len]
        embedded = self.dropout["start_dropout"](self.embeddings(input_batch))
        # embedded.size() = [batch_size, sent_len, embedding_dim]

        embedded = embedded.permute(1, 0, 2)
        # embedded.size() = [sent_len, batch_size, embedding_dim]

        output_1, (_, _) = self.lstm_1(embedded)
        output_1 = self.dropout["middle_dropout"](nn.ReLU()(output_1))
        # output_1.size() = [sent_len, batch_size, hid_dim * num_directions]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]

        output_2, (_, _) = self.lstm_2(output_1)
        output_2 = self.dropout["middle_dropout"](nn.ReLU()(output_2))
        # output_2.size() = [sent_len, batch_size, hid_dim * num_directions]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]

        cat = torch.cat((output_2, output_1, embedded), dim=2).permute(1, 0, 2)
        # cat.size() = [batch_size, sent_len, (2 *hid_dim * num_directions) + embedding_dim]

        conved = [conv(cat).squeeze(3) for conv in self.convs]
        # x = (2 * hid_dim * num_directions) + embedding_dim
        # conved_n.size() = [batch size, n_filters, x - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n.size() = [batch size, n_filters]
        cat_cnn = torch.cat(pooled, dim=1)
        # cat_cnn.size() = [batch size, n_filters * len(filter_sizes)]
        return self.fully_connected_layers(cat_cnn)

