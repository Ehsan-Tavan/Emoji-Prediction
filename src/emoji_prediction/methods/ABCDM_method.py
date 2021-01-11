#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
ABCDM_model.py is a module for ABCDM model
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
__date__ = "01/11/2021"


class ABCDM(nn.Module):
    """
    In this class we implement ABCDM model
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])

        self.lstm = nn.LSTM(kwargs["embedding_dim"],
                            hidden_size=kwargs["lstm_hidden_dim"],
                            num_layers=kwargs["lstm_layers"],
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["dropout"] if kwargs["lstm_layers"] > 1 else 0)

        self.gru = nn.GRU(kwargs["embedding_dim"],
                          hidden_size=kwargs["lstm_hidden_dim"],
                          num_layers=kwargs["lstm_layers"],
                          bidirectional=kwargs["bidirectional"],
                          dropout=kwargs["dropout"] if kwargs["lstm_layers"] > 1 else 0)

        # We will use da = 350, r = 30 & penalization_coeff = 1 as
        # per given in the self-attention original ICLR paper
        self.w_s1 = nn.Linear(in_features=(2 * kwargs["lstm_hidden_dim"]),
                              out_features=350)
        self.w_s2 = nn.Linear(in_features=350, out_features=30)

        self.convs_1 = nn.ModuleList([
            nn.Conv1d(in_channels=2*kwargs["lstm_hidden_dim"],
                      out_channels=kwargs["n_filters"],
                      kernel_size=fs)
            for fs in kwargs["filter_sizes"]
        ])

        self.convs_2 = nn.ModuleList([
            nn.Conv1d(in_channels=2*kwargs["lstm_hidden_dim"],
                      out_channels=kwargs["n_filters"],
                      kernel_size=fs)
            for fs in kwargs["filter_sizes"]
        ])

        self.bn1 = nn.BatchNorm1d(8 * kwargs["n_filters"])

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=8 * kwargs["n_filters"],
                      out_features=kwargs["dense_units"]),

            nn.Linear(in_features=kwargs["dense_units"],
                      out_features=kwargs["output_size"])
        )

        self.dropout = nn.Dropout(kwargs["dropout"])

    def attention_net(self, lstm_output):
        """
        Now we will use self attention mechanism to produce a matrix embedding
        of the input sentence in which every row represents an encoding of the
        inout sentence but giving an attention to a specific part of the sentence.
        We will use 30 such embedding of the input sentence and then finally we will
        concatenate all the 30 sentence embedding vectors and connect it to a fully connected layer
        of size 2000 which will be connected to the output layer of size 2 returning logits for our
        two classes i.e., pos & neg.
        Arguments
        ---------
        lstm_output = A tensor containing hidden states corresponding to each
        time step of the LSTM network.
        ---------
        Returns : Final Attention weight matrix for all the 30 different sentence embedding
        in which each of 30 embeddings give
                  attention to different parts of the input sentence.
        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)
        """
        attn_weight_matrix = self.w_s2(torch.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, sent_len]

        embedded = self.embeddings(input_batch)
        # embedded.size() = [batch_size, sent_len, embedding_dim]
        embedded = embedded.permute(1, 0, 2)
        embedded = self.dropout(embedded)
        # embedded.size() = [sent_len, batch_size, embedding_dim]

        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_output = lstm_output.permute(1, 0, 2)
        lstm_output = self.dropout(lstm_output)
        # lstm_output.size() = [batch_size, sent_len, hid_dim * num_directions]

        lstm_attn_weight_matrix = self.attention_net(lstm_output)
        # lstm_attn_weight_matrix.size() = [batch_size, 30, sent_len]

        lstm_hidden_matrix = torch.bmm(lstm_attn_weight_matrix, lstm_output)
        # lstm_attn_weight_matrix.size() = [batch_size, 30, 2*hid_dim]

        gru_output, hidden = self.gru(embedded)
        # output.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]

        gru_output = gru_output.permute(1, 0, 2)
        gru_output = self.dropout(gru_output)
        # gru_output.size() = [batch_size, sent_len, hid_dim * num_directions]

        gru_attn_weight_matrix = self.attention_net(gru_output)
        # gru_attn_weight_matrix.size() = [batch_size, 30, sent_len]

        gru_hidden_matrix = torch.bmm(gru_attn_weight_matrix, gru_output)
        # gru_hidden_matrix.size() = [batch_size, 30, 2*hid_dim]

        lstm_conved = [self.dropout(conv(lstm_hidden_matrix.permute(0, 2, 1))) for conv in self.convs_1]

        lstm_max_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in lstm_conved]
        lstm_max_pooled = torch.cat(lstm_max_pooled, dim=1)
        # lstm_max_pooled.size() = [batch_size, 2*n_filters]

        lstm_avg_pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in lstm_conved]
        lstm_avg_pooled = torch.cat(lstm_avg_pooled, dim=1)
        # lstm_avg_pooled.size() = [batch_size, 2*n_filters]

        gru_conved = [self.dropout(conv(gru_hidden_matrix.permute(0, 2, 1))) for conv in self.convs_2]

        gru_max_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in gru_conved]
        gru_max_pooled = torch.cat(gru_max_pooled, dim=1)
        # gru_max_pooled.size() = [batch_size, 2*n_filters]

        gru_avg_pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in gru_conved]
        gru_avg_pooled = torch.cat(gru_avg_pooled, dim=1)
        # gru_avg_pooled.size() = [batch_size, 2*n_filters]

        poled_concat = F.relu(self.bn1(torch.cat((lstm_max_pooled,
                                                  lstm_avg_pooled,
                                                  gru_max_pooled,
                                                  gru_avg_pooled),
                                                 dim=1)))
        # poled_concat.size() = [batch_size, 8*n_filters]

        return self.fully_connected_layers(poled_concat)


model = ABCDM(vocab_size=100, embedding_dim=300, pad_idx=1, lstm_hidden_dim=128,
              lstm_layers=1, bidirectional=True, dropout=0.2, n_filters=32, filter_sizes=[4, 6],
              dense_units=64, output_size=15)

text = torch.rand((32, 100))

model.forward(input_batch=text.long())