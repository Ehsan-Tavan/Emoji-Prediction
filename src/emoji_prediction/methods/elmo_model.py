#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
elmo_model.py is a module for elmo model
"""

import torch
from torch import nn
import torch.nn.functional as F
from elmoformanylangs import Embedder
import numpy as np
from emoji_prediction.config.elmo_config import DEVICE


__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/09/2021"


class ELMo(nn.Module):
    """
    In this class we implement elmo model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.elmo_embedding = Embedder(kwargs["elmo_model_path"],
                                       batch_size=kwargs["batch_size"])

        self.lstm = nn.LSTM(input_size=kwargs["elmo_output_dim"],
                            hidden_size=kwargs["lstm_hidden_dim"],
                            num_layers=kwargs["lstm_layers"],
                            dropout=kwargs["dropout"] if kwargs["lstm_layers"] > 1
                            else 0,
                            bidirectional=kwargs["bidirectional"])

        self.output = nn.Linear(in_features=2*kwargs["lstm_hidden_dim"],
                                out_features=kwargs["output_size"])

        self.dropout = nn.Dropout(kwargs["dropout"])

        self.idx2word = kwargs["idx2word"]
        self.pad_idx = kwargs["pad_idx"]

    def elmo_encoder(self, input_batch, text_length):
        input_batch = input_batch.tolist()
        output_batch = list()
        str_sen = list()
        sen_len = len(input_batch[0])
        for sen in input_batch:
            tmp = ""
            for idx in sen:
                if idx == self.pad_idx:
                    break
                tmp = tmp + "**" + self.idx2word[idx]
            str_sen.append(tmp.strip().split("**")[1:])

        elmo_output = self.elmo_embedding.sents2elmo(str_sen)
        for el_out, txt_len in zip(elmo_output, text_length):
            n_pad = sen_len - txt_len
            if n_pad > 0:
                pad = np.zeros((n_pad, 1024))
                final_embedding = np.concatenate((el_out, pad), axis=0)
                output_batch.append(final_embedding)
            else:
                output_batch.append(el_out)

        tensor_output_batch = torch.FloatTensor(output_batch)
        return tensor_output_batch.to(DEVICE)

    def forward(self, input_batch, text_length):
        # input_batch.size() = [batch_size, sent_len]

        elmo_output = self.elmo_encoder(input_batch, text_length)
        # input_batch.size() = [batch_size, sent_len, elmo_out_dim]

        elmo_output = elmo_output.permute(1, 0, 2)
        # input_batch.size() = [sent_len, batch_size, elmo_out_dim]

        output, (hidden, cell) = self.lstm(elmo_output)
        # output_1.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        hidden_concat = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden.size() = [batch_size, hid_dim * num_directions]

        return self.output(hidden_concat)
