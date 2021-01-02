#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
parallel_cnn_lstm_model.py is written for parallel cnn_lstm model
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
__date__ = "01/02/2020"


class ParallelCnnLstm(nn.Module):
    """
    In this class we implement parallel_cnn_lstm model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.embeddings = nn.Embedding(kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=kwargs["n_filters"],
                      kernel_size=(fs, kwargs["embedding_dim"]))
            for fs in kwargs["filter_sizes"]
        ])

        self.lstm = nn.LSTM(input_size=kwargs["embedding_dim"],
                            hidden_size=kwargs["lstm_units"],
                            num_layers=kwargs["lstm_layers"],
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["middle_dropout"] if kwargs["lstm_layers"] > 1 else 0)

        fully_connected_in_features = (len(kwargs["filter_sizes"]) *
                                       kwargs["n_filters"]) + (2 * kwargs["lstm_units"])
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=fully_connected_in_features,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(kwargs["end_dropout"]),
            nn.Linear(in_features=512,
                      out_features=256),
            nn.ReLU(),
            nn.Dropout(kwargs["end_dropout"]),
            nn.Linear(in_features=256,
                      out_features=kwargs["output_size"])
        )

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, sent_len]

        embedded = self.start_dropout(self.embeddings(input_batch))
        # embedded.size() = [batch_size, sent_len, emb_dim]

        embedded_cnn = embedded.unsqueeze(1)
        # embedded_cnn.size() = [batch size, 1, sent len, emb dim]

        embedded_lstm = embedded.permute(1, 0, 2)
        # embedded_lstm.size() = [sent_len, batch_size, emb_dim]

        conved = [self.middle_dropout(nn.ReLU()(conv(embedded_cnn))).squeeze(3) for conv in self.convs]
        # conved_n.size() = [batch_size, n_filters, sent_len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n.size() = [batch_size, n_filters]

        cat_cnn = torch.cat(pooled, dim=1)
        # cat_cnn.size() = [batch_size, n_filters * len(filter_sizes)]

        _, (hidden, _) = self.lstm(embedded_lstm)
        # _ = [sent_len, batch_size, hid_dim * num directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _ = [num_layers * num_directions, batch_size, hid_dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden_concat = self.middle_dropout(F.relu(hidden_concat))
        # hidden.size() = [batch_size, hid_dim * num_directions]

        cat = torch.cat((hidden_concat, cat_cnn), dim=1)
        # hidden.size() = [batch_size, [hid_dim * num_directions] + [n_filters * len(filter_sizes)]

        return self.fully_connected_layers(cat)
