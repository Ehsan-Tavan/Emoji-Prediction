#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
Hybrid_model.py is a module for Hybrid model
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
__date__ = "01/10/2021"


class Hybrid(nn.Module):
    """
    In this class we implement Hybrid model
    """
    def __init__(self, **kwargs):
        super().__init__()

        # Embedding Layer
        self.char_embeddings = nn.Embedding(num_embeddings=kwargs["num_char"],
                                            embedding_dim=kwargs["num_char"] if kwargs["one_hot"]
                                            else kwargs["char_emd_dim"],
                                            padding_idx=kwargs["pad_idx"])

        self.token_embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                             embedding_dim=kwargs["vocab_emb_dim"],
                                             padding_idx=kwargs["pad_idx"])

        self.conv = nn.Conv1d(in_channels=kwargs["vocab_size"] if kwargs["one_hot"]
                              else kwargs["char_emd_dim"],
                              out_channels=kwargs["num_channels"],
                              kernel_size=3)
        self.cnn_btn = nn.BatchNorm1d(kwargs["num_channels"])

        self.lstm = nn.LSTM(input_size=kwargs["vocab_emb_dim"],
                            hidden_size=kwargs["lstm_hid_dim"],
                            bidirectional=kwargs["bidirectional"],
                            num_layers=1)

        self.lstm_btn = nn.BatchNorm1d(2*kwargs["lstm_hid_dim"])

        self.output = nn.Linear(in_features=2 * kwargs["lstm_hid_dim"] + kwargs["num_channels"],
                                out_features=kwargs["n_class"])

        self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self, input_char, input_token):
        # input_char.size() = [batch_size, char_sen_len]
        # input_token.size() = [batch_size, token_sen_len]

        char_embedded = self.char_embeddings(input_char)
        # input_char.size() = [batch_size, char_sen_len, char_emd_dim]

        char_embedded = char_embedded.permute(0, 2, 1)
        # input_char.size() = [batch_size, char_emd_dim, char_sen_len]

        token_embedded = self.token_embeddings(input_token)
        # input_token.size() = [batch_size, token_sen_len, vocab_emb_dim]

        token_embedded = token_embedded.permute(1, 0, 2)
        # input_token.size() = [token_sen_len, batch_size, vocab_emb_dim]

        conv_out = self.conv(char_embedded)
        # conv_out.size() = [batch_size, num_channels, char_sen_len - 2]

        maxpool_out = self.cnn_btn(F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2))
        # maxpool_out.size() = [batch_size, num_channels]

        output, (hidden, cell) = self.lstm(token_embedded)
        # output_1.size() = [token_sen_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        hidden_concat = self.lstm_btn(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden.size() = [batch size, hid dim * num directions]

        finall_feature = torch.cat((maxpool_out, hidden_concat), dim=1)
        # output_feature.size() = [batch size, hid dim * num directions + num_channels]

        return self.output(finall_feature)
