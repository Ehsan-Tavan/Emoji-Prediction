#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
transformer_model.py is written for transformer model
"""

import torch
from torch import nn

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/17/2021"


class Transformer(nn.Module):
    """
    In this class we implement encoder of transformer
    """
    def __init__(self,
                 hid_dim,
                 final_dropout,
                 output_size,
                 encoder,
                 src_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.device = device

        self.fully_connected_layers = nn.Linear(
            in_features=hid_dim, out_features=output_size
        )

    def make_input_mask(self, input_batch):
        # input_batch.size() = [batch_size, input_len]
        input_mask = (input_batch != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # input_mask = [batch_size, 1, 1, input_len]

        return input_mask

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, input_len]

        input_mask = self.make_input_mask(input_batch)
        # input_mask.size() = [batch_size, 1, 1, input_len]

        enc_output = self.encoder(input_batch, input_mask)
        # enc_output.size() = [batch_size, input_len, hid_dim]

        enc_output = enc_output.permute(0, 2, 1)
        # enc_output.size() = [batch_size, hid_dim, input_len]

        enc_output = nn.MaxPool1d(enc_output.size()[2])(enc_output).squeeze(2)
        # enc_input.size() = [batch_size, hid_dim]

        # enc_output = torch.flatten(enc_output, start_dim=1)
        # # enc_input.size() = [batch_size, input_len * hid_dim]

        return self.fully_connected_layers(enc_output)


class Encoder(nn.Module):
    """
    In this class we implement encoder of transformer
    """
    def __init__(self, vocab_size,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()
        self.tok_embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=hid_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=max_length,
                                          embedding_dim=hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, input_batch, input_mask):
        # input_batch.size() = [batch_size, input_len]
        # input_mask.size() = [batch_size, 1, 1, input_len]

        batch_size = input_batch.shape[0]
        input_len = input_batch.shape[1]

        pos = torch.arange(0, input_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos.size() = [batch_size, input_len]

        input_batch = self.dropout((self.tok_embedding(input_batch) * self.scale) + self.pos_embedding(pos))
        # input_batch.size() = [batch_size, input_len, hid_dim]

        for layer in self.layers:
            input_batch = layer(input_batch, input_mask)
        # input_batch.size() = [batch_size, input_len, hid_dim]

        return input_batch


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch, input_mask):
        # input_batch.size() = [batch_size, input_len, hid_dim]
        # input_mask.size() = [batch_size, input_len]

        # self attention
        _input_batch, _ = self.self_attention(input_batch, input_batch, input_batch, input_mask)

        # dropout, residual connection and layer norm
        # # input_batch = self.self_attn_layer_norm(input_batch + self.dropout(_input_batch))
        # input_batch.size() = [batch_size, input_len, hid_dim]

        # positionwise feedforward
        # # _input_batch = self.positionwise_feedforward(input_batch)
        _input_batch = self.positionwise_feedforward(input_batch + self.dropout(_input_batch))

        # dropout, residual and layer norm
        # # input_batch = self.ff_layer_norm(input_batch + self.dropout(_input_batch))
        # input_batch.size() = [batch_size, input_len, hid_dim]

        return input_batch + self.dropout(_input_batch)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query.size() = [batch_size, query_len, hid_dim]
        # key.size() = [batch_size, key_len, hid_dim]
        # value.size() = [batch_size, value_len, hid_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # query.size() = [batch_size, query_len, hid_dim]
        # key.size() = [batch_size, key_len, hid_dim]
        # value.size() = [batch_size, value_len, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q.size() = [batch_size, n_heads, query_len, head_dim]
        # K.size() = [batch_size, n_heads, query_len, head_dim]
        # V.size() = [batch_size, n_heads, query_len, head_dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy.size() = [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention.size() = [batch_size, n_heads, query_len, key_len]

        x = torch.matmul(self.dropout(attention), V)

        # x.size() = [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x.size() = [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x.size() = [batch_size, query_len, hid_dim]

        x = self.fc_o(x)

        # x.size() = [batch_size, query_len, hid_dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.size() = [batch_size, seq_len, hid_dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x.size() = [batch_size, seq_len, pf_dim]

        x = self.fc_2(x)

        # x.size() = [batch_size, seq_len, hid_dim]

        return x


# # create model
#
# HID_DIM = 256
# ENC_LAYERS = 3
# ENC_HEADS = 8
# ENC_PF_DIM = 512
# ENC_DROPOUT = 0.1
# FINAL_DROPOUT = 0.3
# MAX_LENGTH = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# enc = Encoder(vocab_size=100,
#               hid_dim=HID_DIM,
#               n_layers=ENC_LAYERS,
#               n_heads=ENC_HEADS,
#               pf_dim=ENC_PF_DIM,
#               dropout=ENC_DROPOUT,
#               device=DEVICE)
#
# model = Transformer(hid_dim=HID_DIM, final_dropout=FINAL_DROPOUT, encoder=enc,
#                     output_size=15, device=DEVICE,
#                     src_pad_idx=1)
#
# text = torch.rand((20, 100))
#
# model.forward(input_batch=text.long())