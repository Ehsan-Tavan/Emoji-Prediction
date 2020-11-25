#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
deepmoji_model.py is written for deepmoji model
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
__date__ = "11/25/2020"


class DeeoMoji(nn.Module):
    """
    In this class we implement deepmoji model
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

        # We will use da = 350, r = 30 & penalization_coeff = 1 as
        # per given in the self-attention original ICLR paper
        self.w_s1 = nn.Linear(in_features=(2 * 2 * kwargs["lstm_hidden_dim"]) +
                              kwargs["embedding_dim"],
                              out_features=350)
        self.w_s2 = nn.Linear(in_features=350, out_features=1)

        self.output = nn.Linear(in_features=((2 * 2 * kwargs["lstm_hidden_dim"]) +
                                             kwargs["embedding_dim"]),
                                out_features=kwargs["output_size"])

        self.dropout = {"start_dropout": nn.Dropout(kwargs["start_dropout"]),
                        "final_dropout": nn.Dropout(kwargs["final_dropout"])}

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

    def forward(self, input_batch, text_length, pack_padded=False):
        # input_batch.size() = [batch_size, sent_len]

        if pack_padded:
            # sort input_batch sentences by their length
            sorted_text_len, perm_idx = text_length.sort(descending=True)
            input_batch = input_batch[perm_idx]
            _, recover_idx = perm_idx.sort(descending=False)

        embedded = self.embeddings(input_batch)
        embedded = torch.tanh(embedded)
        embedded = self.dropout["start_dropout"](embedded)
        # embedded.size() = [batch_size, sent_len, embedding_dim]

        embedded = embedded.permute(1, 0, 2)
        # embedded.size() = [sent_len, batch_size, embedding_dim]

        # pack padded sequence
        if pack_padded:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                         lengths=sorted_text_len.tolist(),
                                                         batch_first=True)

        output_1, (_, _) = self.lstm_1(embedded)
        # output_1.size() = [sent_len, batch_size, hid_dim * num_directions]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]

        output_2, (_, _) = self.lstm_2(output_1)
        # output_2.size() = [sent_len, batch_size, hid_dim * num_directions]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]

        all_features = torch.cat((output_2, output_1, embedded), dim=2).permute(1, 0, 2)
        # all_features.size() = [batch_size, sent_len,
        # (2 * hid_dim * num_directions) + embedding_dim]

        attn_weight_matrix = self.attention_net(all_features)
        # attn_weight_matrix.size() = [batch_size, 1, sent_len]

        hidden_matrix = torch.bmm(attn_weight_matrix, all_features)
        # hidden_matrix.size() = [batch_size, 1, (2 * hid_dim * num_directions) + embedding_dim]

        # unpack sequence
        if pack_padded:
            hidden_matrix, _ = nn.utils.rnn.pad_packed_sequence(hidden_matrix)

        hidden_matrix = self.dropout["final_dropout"](hidden_matrix.squeeze(1))
        print(hidden_matrix.size())

        # hidden_matrix = self.dropout["final_dropout"](
        #     hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2])
        # )
        # hidden_matrix.size() = [batch_size, r * ((2 * hid_dim * num_directions) + embedding_dim)]

        pred = self.output(hidden_matrix).permute(1, 0, 2)
        print(pred.size())
        if pack_padded:
            pred = pred[recover_idx]
        return pred
