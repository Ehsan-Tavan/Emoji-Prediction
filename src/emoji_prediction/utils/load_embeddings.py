#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
load_embeddings.py is writen for load pre_train embedding file
"""

import numpy as np
import pickle as pkl
from emoji_prediction.config.cnn_config import WORD2IDX_PATH


class LoadEmbeddings:
    """
    In this class we load and save pre_train embeddings
    """
    def __init__(self, embedding_text_path):
        self.embedding_text_path = embedding_text_path

    def load_embedding_file(self):
        embedding_file = open(self.embedding_text_path, "r", encoding="utf-8")
        return embedding_file

    @staticmethod
    def load_word2idx():
        with open(WORD2IDX_PATH, "rb") as f:
            word2idx = pkl.load(f)
        return word2idx

    def create_embedding_matrix(self, word2idx, embedding_file, embedding_dim):
        matrix_len = len(list(word2idx.keys()))
        weights_matrix = np.zeros((matrix_len, embedding_dim))
        words_found = 0
        for line in embedding_file:
            line = line.split()
