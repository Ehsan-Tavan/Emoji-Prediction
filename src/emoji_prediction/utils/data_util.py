#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
data_util.py is writen for creating iterator and save field
"""

import logging
import hazm
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from torchtext import data
from torchtext.vocab import Vectors
from sklearn.utils import class_weight

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/08/2021"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class DataSet:
    """
    DataSet Class use for preparing data
    and iterator for training model.
    """
    def __init__(self, **kwargs):
        self.files_address = {
            "train_data_path": kwargs["train_data_path"],
            "validation_data_path": kwargs["validation_data_path"],
            "test_data_path": kwargs["test_data_path"],
            "embedding_path": kwargs["embedding_path"],
            "word_emotion_path": kwargs["word_emotion_path"]
        }

        self.class_weight = None
        self.text_field = None
        self.emotion_matrix = None

        self.iterator_dict = dict()
        self.embedding_dict = dict()
        self.unk_idx_dict = dict()
        self.pad_idx_dict = dict()
        self.num_vocab_dict = dict()

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
    def create_fields(sen_max_len=None):
        """
        This method is writen for creating torchtext fields
        :param sen_max_len: maximum length of sentence
        :return:
            dictionary_fields: dictionary of data fields
            data_fields: list of data fields
        """
        # Create Field for data
        if sen_max_len is not None:
            text_field = data.Field(tokenize=hazm.word_tokenize, batch_first=True, include_lengths=True,
                                    fix_length=sen_max_len)
        else:
            text_field = data.Field(tokenize=hazm.word_tokenize, batch_first=True, include_lengths=True)

        label_field = data.LabelField()
        dictionary_fields = {
            "text_field": text_field,
            "label_field": label_field
        }

        # create list of data fields
        data_fields = [("text", text_field), ("label", label_field)]
        return dictionary_fields, data_fields

    @staticmethod
    def create_emotion_embedding(word_emotion_path, text_field, embedding_dim):
        """
        create_emotion_embedding method is written for create emotional embedding matrix
        :param word_emotion_path: address of emotional dictionary
        :param text_field: text_field
        :param embedding_dim: dimension of emotional embedding
        :return:
            weight_matrix: emotional embedding
        """
        lematizer = hazm.Lemmatizer()
        # load pickle dictionary of emotional embedding
        with open(word_emotion_path, "rb") as file:
            word_emotion_dict = pkl.load(file)

        # create weight_matrix as zero matrix
        weight_matrix = np.zeros((len(text_field.vocab), embedding_dim))

        for token, idx in text_field.vocab.stoi.items():
            lemma_token = lematizer.lemmatize(token)
            emotion_embedding = word_emotion_dict.get(lemma_token)
            if emotion_embedding is not None:
                weight_matrix[idx] = emotion_embedding
        return weight_matrix

    def load_data(self, text_field_path, label_field_path, device, batch_size, sen_max_len=None,
                  use_emotion=False, emotion_embedding_dim=10, min_freq=20):
        """
        load_data method is written for creating iterator for train and test data
        :param text_field_path: path for text_field
        :param label_field_path: path for label_field
        :param device: gpu or cpu
        :param batch_size: number of sample in batch
        :param sen_max_len: maximum length of sentence
        :param use_emotion: if true model use emotion feature of words
        :param emotion_embedding_dim: dimension of emotion embedding
        :param min_freq: min frequency of words
        """
        # create fields
        logging.info("Start creating fields.")
        dictionary_fields, data_fields = self.create_fields(sen_max_len)

        # Load data from pd.DataFrame into torchtext.data.Dataset
        logging.info("Start creating train example.")
        train_examples = [data.Example.fromlist(row, data_fields) for row in
                          self.read_csv_file(self.files_address["train_data_path"]).values.tolist()]

        train_data = data.Dataset(train_examples, data_fields)

        logging.info("Start creating validation example.")
        validation_examples = [data.Example.fromlist(row, data_fields) for row in
                               self.read_csv_file(self.files_address["validation_data_path"]).values.tolist()]

        validation_data = data.Dataset(validation_examples, data_fields)

        logging.info("Start creating test example.")
        test_examples = [data.Example.fromlist(row, data_fields) for row in
                         self.read_csv_file(self.files_address["test_data_path"]).values.tolist()]
        test_data = data.Dataset(test_examples, data_fields)

        # build vocab in all fields
        logging.info("Start creating text_field vocabs.")
        dictionary_fields["text_field"].build_vocab(train_data,
                                                    min_freq=min_freq,
                                                    unk_init=torch.Tensor.normal_,
                                                    vectors=Vectors(
                                                        self.files_address["embedding_path"]
                                                    ))

        # create emotion matrix
        if use_emotion:
            self.emotion_matrix = self.create_emotion_embedding(self.files_address["word_emotion_path"],
                                                                dictionary_fields["text_field"],
                                                                emotion_embedding_dim)
            logging.info("Create emotion_matrix")

        self.text_field = dictionary_fields["text_field"]
        logging.info("Start creating label_field vocabs.")
        dictionary_fields["label_field"].build_vocab(train_data)

        # get embedding vector for all vocabs
        self.embedding_dict["vocab_embedding_vectors"] = \
            dictionary_fields["text_field"].vocab.vectors

        # count number of unique vocab in all fields
        self.num_vocab_dict = self.calculate_num_vocabs(dictionary_fields)

        # get pad index in all fields
        self.pad_idx_dict = self.find_pad_index(dictionary_fields)

        # get unk index in all fields
        self.unk_idx_dict = self.find_unk_index(dictionary_fields)

        # calculate class weight for handling imbalanced data
        self.class_weight = self.calculate_class_weight(dictionary_fields)

        # saving fields
        self.save_fields(dictionary_fields, text_field_path, label_field_path)

        # creating iterators for training model
        logging.info("Start creating iterator.")
        self.iterator_dict = self.creating_iterator(train_data=train_data,
                                                    valid_data=validation_data,
                                                    test_data=test_data,
                                                    batch_size=batch_size,
                                                    device=device)

        logging.info("Loaded %d train examples", len(train_data))
        logging.info("Loaded %d valid examples", len(validation_data))
        logging.info("Loaded %d test examples", len(test_data))

    @staticmethod
    def save_fields(dictionary_fields, text_field_path, label_field_path):
        """
        save_fields method is writen for saving fields
        :param dictionary_fields: dictionary of fields
        :param text_field_path: path for text_field
        :param label_field_path: path for label_field
        """
        logging.info("Start saving fields...")
        # save text_field
        torch.save(dictionary_fields["text_field"], text_field_path)
        logging.info("text_field is saved.")

        # save label_field
        torch.save(dictionary_fields["label_field"], label_field_path)
        logging.info("label_field is saved.")

    @staticmethod
    def calculate_class_weight(dictionary_fields):
        """
        calculate_class_weight method is written for calculate class weight
        :param dictionary_fields: dictionary of fields
        :return:
            class_weights: calculated class weight
        """
        label_list = []
        for label, idx in dictionary_fields["label_field"].vocab.stoi.items():
            for _ in range(dictionary_fields["label_field"].vocab.freqs[label]):
                label_list.append(idx)
        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                          classes=np.unique(label_list),
                                                          y=label_list).astype(np.float32)
        return class_weights

    @staticmethod
    def creating_iterator(**kwargs):
        """
        creating_iterator method is written for create iterator for training model
        :param kwargs:
            train_data: train dataSet
            valid_data: validation dataSet
            test_data: test dataSet
            batch_size: number of sample in batch
            device: gpu or cpu

        :return:
            iterator_dict: dictionary of iterators
        """
        iterator_dict = {
            "train_iterator": data.BucketIterator(kwargs["train_data"],
                                                  batch_size=kwargs["batch_size"],
                                                  sort=False,
                                                  shuffle=True,
                                                  device=kwargs["device"]),
            "valid_iterator": data.BucketIterator(kwargs["valid_data"],
                                                  batch_size=kwargs["batch_size"],
                                                  sort=False,
                                                  shuffle=True,
                                                  device=kwargs["device"]),
            "test_iterator": data.BucketIterator(kwargs["test_data"],
                                                 batch_size=kwargs["batch_size"],
                                                 sort=False,
                                                 shuffle=True,
                                                 device=kwargs["device"])
        }
        return iterator_dict

    @staticmethod
    def calculate_num_vocabs(dictionary_fields):
        """
        calculate_num_vocabs method is written for calculate vocab counts in each field
        :param dictionary_fields: dictionary of fields
        :return:
            num_vocab_dict:  dictionary of vocab counts in each field
        """
        num_vocab_dict = dict()
        num_vocab_dict["num_token"] = len(dictionary_fields["text_field"].vocab)
        num_vocab_dict["num_label"] = len(dictionary_fields["label_field"].vocab)
        return num_vocab_dict

    @staticmethod
    def find_pad_index(dictionary_fields):
        """
        find_pad_index method is written for find pad index in each field
        :param dictionary_fields: dictionary of fields
        :return:
            pad_idx_dict: dictionary of pad index in each field
        """
        pad_idx_dict = dict()
        pad_idx_dict["token_pad_idx"] = dictionary_fields["text_field"]\
            .vocab.stoi[dictionary_fields["text_field"].pad_token]
        return pad_idx_dict

    @staticmethod
    def find_unk_index(dictionary_fields):
        """
        find_unk_index method is written for find unk index in each field
        :param dictionary_fields: dictionary of fields
        :return:
            unk_idx_dict: dictionary of unk index in each field
        """
        unk_idx_dict = dict()
        unk_idx_dict["token_unk_idx"] = dictionary_fields["text_field"] \
            .vocab.stoi[dictionary_fields["text_field"].unk_token]
        return unk_idx_dict


def init_weights(model):
    """
    init_weights method is written for initialize model parameters
    :param model: input model
    """
    for _, param in model.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.1)


def initialize_weights_xavier_uniform(model):
    """
    init_weights method is written for initialize model parameters
    :param model: input model
    """
    if hasattr(model, "weight") and model.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(model.weight.data)
