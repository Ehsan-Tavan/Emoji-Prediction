#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
aug_data_util.py is writen for creating dataLoader for augmentation data
"""

import logging
import hazm
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import class_weight
from emoji_prediction.utils.augmentation import Augmentation
from emoji_prediction.config.cnn_config import BATCH_SIZE, TEXT_FIELD_PATH,\
    LABEL_FIELD_PATH, DEVICE, TRAIN_NORMAL_DATA_PATH, WORD2IDX_PATH, IDX2WORD_PATH,\
    INDEXED_SEN_PATH, INDEXED_LABEL_PATH

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "11/09/2020"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class CreateSamples:
    """
    In this class we create train, validation and test data
    """
    def __init__(self, train_data_path):
        self.train_data_path = train_data_path

    @staticmethod
    def read_csv_file(input_path):
        """
        read_csv_file method is written for reading input csv file
        :param input_path: csv file path
        :return:
            input_df: dataFrame of input data
        """
        input_df = pd.read_csv(input_path)
        input_df = input_df.astype({"tweets": "str"})
        input_df = input_df.astype({"emojis": "str"})
        return input_df[:500]

    @staticmethod
    def create_word_index(train_data, min_count=5):
        """
        create_word_index is written for create word index
        :param train_data: train samples
        :param min_count: min word repetitions
        :return:
            word2idx: word to index dictionary
            idx2word: index to word dictionary
        """
        tokens = set()  # for all unique tokens
        token_counter = Counter()   # To count the number of repetitions of tokens

        for sentence in train_data:
            # tokenizing sentence with hazm
            tokenized_sentence = hazm.word_tokenize(str(sentence))
            tokens.update(tokenized_sentence)
            token_counter.update(tokenized_sentence)

        word2idx = dict()
        idx2word = dict()

        word2idx["<pad>"] = 0
        idx2word[0] = "<pad>"
        word2idx["<unk>"] = 1
        idx2word[1] = "<unk>"

        for token in tokens:
            if token_counter[token] > min_count:
                word2idx[token] = len(word2idx)
                idx2word[len(idx2word)] = token

        return word2idx, idx2word

    @staticmethod
    def save_pickle_data(input_data, path):
        """
        save_pickle_data method is written for save data
        :param input_data: input data
        :param path: path to save pickle
        """
        with open(path, "wb") as pkl_data:
            pkl.dump(input_data, pkl_data)

    @staticmethod
    def create_sentence_idx(text, label, word2idx):
        """
        create_sentence_idx is written to create sentence indexed
        :param text: input sentence
        :param label: input_label
        :param word2idx: word to index dictionary
        :return:
            indexed_sentence:
            indexed_sentence
        """
        indexed_sentence = []
        indexed_label = []
        for sentence, lbl in zip(text, label):
            indexed_label.append(lbl)
            x_tmp = []
            for token in hazm.word_tokenize(str(sentence)):
                try:
                    x_tmp.append(word2idx[token])
                except:
                    x_tmp.append(word2idx["<unk>"])
            indexed_sentence.append(x_tmp)
        max_len = max(len(x) for x in indexed_sentence)
        print(f"max sequence length is {max_len}.")
        return indexed_sentence, indexed_label

    def __run__(self):
        data_frame = self.read_csv_file(self.train_data_path)
        word2idx, idx2word = self.create_word_index(data_frame["tweets"])
        self.save_pickle_data(word2idx, WORD2IDX_PATH)
        self.save_pickle_data(idx2word, IDX2WORD_PATH)
        indexed_sentence, indexed_label = self.create_sentence_idx(data_frame["tweets"],
                                                                   data_frame["emojis"],
                                                                   word2idx)

        self.save_pickle_data(indexed_sentence, INDEXED_SEN_PATH)
        self.save_pickle_data(indexed_label, INDEXED_LABEL_PATH)


class MyDataset(Dataset):
    def __init__(self, texts, labels, max_length=38):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.texts)
        self.augmentation_class = Augmentation(word2idx_pickle_path=WORD2IDX_PATH,
                                               idx2word_pickle_path=IDX2WORD_PATH)
        self.augmentation_methods = {
            "delete_randomly": True,
            "replace_similar_words": True,
            "swap_token": True
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        _text = self.texts[index]
        _label = self.labels[index]

        augment_list = self.augmentation_class.__run__(list(_text),
                                                       self.augmentation_methods)
        data_list, label_list = [], []
        for seq in augment_list:
            if len(seq) >= self.max_length:
                seq = list(seq)[:self.max_length]

            elif len(seq) < self.max_length:
                seq = list(seq) + ([0] * (self.max_length - len(seq)))

            data_list.append(seq)
            label_list.append(_label)
        print(data_list)
        print(label_list)
        return np.array(data_list), np.array(label_list)


if __name__ == "__main__":
    mclass = CreateSamples(train_data_path=TRAIN_NORMAL_DATA_PATH)
    mclass.__run__()

    with open(INDEXED_SEN_PATH, "rb") as f:
        x = pkl.load(f)
    with open(INDEXED_LABEL_PATH, "rb") as f:
        y = pkl.load(f)

    dataset = MyDataset(np.array(x), np.array(y))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    for data in dataloader:
        print("ok")
        print(data)
