#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
DCBiLstm_model.py is written for DC_BiLstm model
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


class DCBiLstm(nn.Module):
    """
    In this class we implement DC_BiLstm model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.use_emotion = kwargs["use_emotion"]
        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])

        if self.use_emotion:
            self.emotion_embeddings = nn.Embedding(
                num_embeddings=kwargs["vocab_size"],
                embedding_dim=kwargs["emotion_embedding_dim"],
                padding_idx=kwargs["pad_idx"]
            )

        self.lstm_1 = nn.LSTM(input_size=kwargs["embedding_dim"] +
                              kwargs["emotion_embedding_dim"] if self.use_emotion
                              else kwargs["embedding_dim"],
                              hidden_size=kwargs["lstm_hidden_dim"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.lstm_2 = nn.LSTM(input_size=2 * kwargs["lstm_hidden_dim"] +
                              kwargs["embedding_dim"] +
                              kwargs["emotion_embedding_dim"] if self.use_emotion
                              else 2 * kwargs["lstm_hidden_dim"] +
                              kwargs["embedding_dim"],
                              hidden_size=kwargs["lstm_hidden_dim"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.lstm_3 = nn.LSTM(input_size=4 * kwargs["lstm_hidden_dim"] +
                              kwargs["embedding_dim"] +
                              kwargs["emotion_embedding_dim"] if self.use_emotion
                              else 4 * kwargs["lstm_hidden_dim"] +
                              kwargs["embedding_dim"],
                              hidden_size=kwargs["lstm_hidden_dim"],
                              num_layers=1,
                              bidirectional=kwargs["bidirectional"])

        self.output = nn.Linear(in_features=2*kwargs["lstm_hidden_dim"],
                                out_features=kwargs["output_size"])

        self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, sent_len]

        embedded = self.embeddings(input_batch)
        # embedded.size() = [batch_size, sent_len, embedding_dim]

        if self.use_emotion:
            emotion_embedded = self.emotion_embeddings(input_batch)
            embedded = torch.cat((embedded, emotion_embedded), dim=2)

        embedded = embedded.permute(1, 0, 2)
        embedded = self.dropout(embedded)
        # embedded.size() = [sent_len, batch_size, embedding_dim]

        output_1, (hidden, cell) = self.lstm_1(embedded)
        # output_1.size() = [sent_len, batch_size, hid_dim*num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_2_input = torch.cat((output_1, embedded), dim=2)
        # output_1.size() = [sent_len, batch_size, hid_dim*num_directions+embedding_dim]

        output_2, (hidden, cell) = self.lstm_2(lstm_2_input, (hidden, cell))
        # output_2.size() = [sent_len, batch_size, hid_dim*num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_3_input = torch.cat((output_1, output_2, embedded), dim=2)
        # output_1.size() = [sent_len, batch_size, 2*hid_dim*num_directions+embedding_dim]

        output_3, (_, _) = self.lstm_3(lstm_3_input, (hidden, cell))
        # output_3.size() = [sent_len, batch_size, hid_dim * num_directions]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]

        output_3 = output_3.permute(1, 2, 0)
        # output_3.size() = [batch_size, sent_len, hid_dim * num_directions]

        # avg_pooling = nn.AdaptiveAvgPool2d((output_3.size()[1], 1))(output_3)
        avg_pooling = F.avg_pool1d(output_3, output_3.shape[2]).squeeze(2)
        # avg_pooling.size() = [batch_size, sent_len]

        avg_pooling = self.dropout(avg_pooling)
        # avg_pooling.size() = [batch_size, sent_len]

        return self.output(avg_pooling)
