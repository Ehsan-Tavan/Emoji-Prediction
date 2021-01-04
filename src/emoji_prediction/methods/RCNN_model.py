#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
RCNN_model.py is a module for RCNN model
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
__date__ = "01/04/2021"


class RCNN(nn.Module):
    """
    In this class we implement RCNN model
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.use_emotion = kwargs["use_emotion"]

        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])

        self.lstm = nn.LSTM(input_size=kwargs["embedding_dim"],
                            hidden_size=kwargs["lstm_hid_dim"],
                            num_layers=kwargs["lstm_layers"],
                            dropout=kwargs["dropout"] if kwargs["lstm_layers"] > 1
                            else 0,
                            bidirectional=True)

        self.dropout = nn.Dropout(kwargs["dropout"])

        self.linear = nn.Linear(
            in_features=kwargs["embedding_dim"] + 2 * kwargs["lstm_hid_dim"],
            out_features=kwargs["linear_units"]
        )

        self.tanh = nn.Tanh()

        self.output = nn.Linear(in_features=kwargs["linear_units"],
                                out_features=kwargs["output_size"])

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, sent_len]

        embedded = self.embeddings(input_batch)
        # embedded.size() = [batch_size, sent_len, embedding_dim]

        if self.use_emotion:
            emotion_embedded = self.emotion_embeddings(input_batch)
            embedded = torch.cat((embedded, emotion_embedded), dim=2)

        embedded = embedded.permute(1, 0, 2)
        # embedded.size() = [sent_len, batch_size, embedding_dim]

        lstm_output, (hidden, cell) = self.lstm(embedded)
        # output_1.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        input_features = torch.cat([lstm_output, embedded], 2).permute(1, 0, 2)
        # final_features.size() = [batch_size, sent_len, embedding_dim+2*hid_dim]

        linear_output = self.tanh(self.linear(input_features))
        # linear_output.size() = [batch_size, sent_len, linear_units]

        linear_output = linear_output.permute(0, 2, 1)  # Reshaping fot max_pool
        # linear_output.size() = [batch_size, linear_units, sent_len]

        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        max_out_features = self.dropout(max_out_features)
        # max_out_features.size() = [batch_size, linear_units]

        return self.output(max_out_features)
