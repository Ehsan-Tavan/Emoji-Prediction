#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error
# pylint: disable-msg=no-member

"""
bert_data_util.py is writen for creating iterator and save field
"""

import logging
import torch
import pandas as pd
import functools
from transformers import AutoConfig, AutoTokenizer
from torchtext import data
from emoji_prediction.config.bert_config import DEVICE, BATCH_SIZE, SEN_LEN, TRAIN_AUGMENTATION

__author__ = "Ehsan Tavan"
__project__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "11/28/2020"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class DataSet:
    """
    DataSet Class use for preparing data
    and iterator for training model.
    """
    def __init__(self, **kwargs):
        self.files_address = {"train_data_path": kwargs["train_data_path"],
                              "validation_data_path": kwargs["validation_data_path"],
                              "test_data_path": kwargs["test_data_path"]}

        self.iterator_dict = dict()
        self.pad_idx_dict = dict()
        self.num_vocab_dict = dict()
        self.dictionary_fields = dict()

        self.vocabs, self.word2idx, self.idx2word = None, None, None

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
    def load_bert_tokenizer(model_config, dl=False, save=True):
        """
        load_bert_tokenizer method is written for download and save bert tokenizer
        :param model_config: model_config
        :param dl: if true download preTrain model else load from local
        :param save: if true save preTrain model
        :return:
            tokenizer: bert tokenizer
        """
        if dl:
            # download tokenizer
            config = AutoConfig.from_pretrained(model_config["url_path"])
            tokenizer = AutoTokenizer.from_pretrained(model_config["url_path"])
            if save:
                # save tokenizer
                tokenizer.save_pretrained(model_config["tokenizer_path"])
                config.save_pretrained(model_config["tokenizer_path"])
        else:
            # load from local
            tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
            config = AutoConfig.from_pretrained(model_config["tokenizer_path"])

        return tokenizer

    @staticmethod
    def tokenize_and_cut(sentence, tokenizer, max_input_length=512):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length - 2]
        return tokens

    def create_vocab_dictionary(self, tokenizer):
        """
        create_vocab_dictionary method is written for create
        required variables for online augmentation
        :param tokenizer: bert tokenizer
        :return:
            vocabs: list of all Common words in our data and bert model
            word2idx: word to index dictionary
            idx2word: index to word dictionary
        """
        vocabs = list()
        word2idx = dict()
        idx2word = dict()
        input_df = self.read_csv_file(self.files_address["validation_data_path"])
        for tweet in input_df.tweet:
            for token in tokenizer.tokenize(tweet):
                vocabs.append(token)
                word2idx[token] = tokenizer.convert_tokens_to_ids(token)
                idx2word[tokenizer.convert_tokens_to_ids(token)] = token
        return vocabs, word2idx, idx2word

    @staticmethod
    def bert_special_tokens(tokenizer):
        """
        bert_special_tokens method is written for get bert special tokens and id's
        :param tokenizer: bert tokenizer
        :return:
        """
        init_token = tokenizer.cls_token
        eos_token = tokenizer.sep_token
        pad_token = tokenizer.pad_token
        unk_token = tokenizer.unk_token

        init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
        eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
        pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
        unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
        return init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx

    def create_fields(self, tokenizer, sen_len):
        """
        This method is writen for creating torchtext fields
        :return: dictionary_fields, data_fields
        """
        init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx =\
            self.bert_special_tokens(tokenizer=tokenizer)

        custom_tokenizer = functools.partial(self.tokenize_and_cut,
                                             tokenizer=tokenizer)

        text_field = data.Field(batch_first=True,
                                include_lengths=True,
                                fix_length=sen_len,
                                use_vocab=False,
                                tokenize=custom_tokenizer,
                                preprocessing=tokenizer.convert_tokens_to_ids,
                                init_token=init_token_idx,
                                eos_token=eos_token_idx,
                                pad_token=pad_token_idx,
                                unk_token=unk_token_idx)

        label_field = data.LabelField()

        dictionary_fields = {"text_field": text_field,
                             "label_field": label_field}

        data_fields = (("text", text_field), ("label", label_field))
        return dictionary_fields, data_fields

    def load_data(self, model_config):
        """
        Create iterator for train and test data
        :param model_config: model config
        """
        # load bert tokenizer
        bert_tokenizer = self.load_bert_tokenizer(model_config)

        # create fields
        logging.info("Start creating fields.")
        self.dictionary_fields, data_fields = self.create_fields(bert_tokenizer, sen_len=SEN_LEN)

        logging.info("Start creating train example.")
        train_examples = [data.Example.fromlist(i, data_fields) for i in
                          self.read_csv_file(
                              self.files_address["train_data_path"]).values.tolist()]
        train_data = data.Dataset(train_examples, data_fields)

        logging.info("Start creating validation example.")
        validation_examples = [data.Example.fromlist(i, data_fields) for i in
                               self.read_csv_file(self.files_address["validation_data_path"]).values.tolist()]
        validation_data = data.Dataset(validation_examples, data_fields)

        logging.info("Start creating test example.")
        test_examples = [data.Example.fromlist(i, data_fields) for i in
                         self.read_csv_file(self.files_address["test_data_path"]).values.tolist()]
        test_data = data.Dataset(test_examples, data_fields)

        logging.info("Start creating label_field vocabs.")
        self.dictionary_fields["label_field"].build_vocab(train_data)

        # count number of unique vocab in all fields
        self.num_vocab_dict = self.calculate_num_vocabs(self.dictionary_fields)

        # saving fields
        self.save_fields(self.dictionary_fields, model_config)

        # creating iterators for training model
        logging.info("Start creating iterator.")
        self.iterator_dict = self.creating_iterator(train_data=train_data,
                                                    validation_data=validation_data,
                                                    test_data=test_data)

        # create required variables for online augmentation
        if TRAIN_AUGMENTATION:
            self.vocabs, self.word2idx, self.idx2word = self.create_vocab_dictionary(bert_tokenizer)

        logging.info("Loaded %d train examples", len(train_data))
        logging.info("Loaded %d validation examples", len(validation_data))
        logging.info("Loaded %d test examples", len(test_data))

    @staticmethod
    def creating_iterator(**kwargs):
        """
        This method create iterator for training model
        :param kwargs:
            train_data: train dataSet
            valid_data: validation dataSet
            test_data: test dataSet
            human_test_data: human test dataSet
        :return:
            iterator_dict: dictionary of iterators
        """
        iterator_dict = {"train_iterator": data.BucketIterator(kwargs["train_data"],
                                                               batch_size=BATCH_SIZE,
                                                               sort=False,
                                                               shuffle=True,
                                                               device=DEVICE),
                         "validation_iterator": data.BucketIterator(kwargs["validation_data"],
                                                                    batch_size=BATCH_SIZE,
                                                                    sort=False,
                                                                    shuffle=True,
                                                                    device=DEVICE),
                         "test_iterator": data.BucketIterator(kwargs["test_data"],
                                                              batch_size=BATCH_SIZE,
                                                              sort=False,
                                                              shuffle=True,
                                                              device=DEVICE)}
        return iterator_dict

    @staticmethod
    def save_fields(dictionary_fields, model_config):
        """
        This method is writen for saving fields
        :param dictionary_fields: dictionary of fields
        :param model_config: model_config
        """
        logging.info("Start saving fields...")

        torch.save(dictionary_fields["text_field"], model_config["text_field_path"])
        logging.info("text_field is saved.")

        torch.save(dictionary_fields["label_field"], model_config["label_field_path"])
        logging.info("label_field is saved.")

    @staticmethod
    def calculate_num_vocabs(dictionary_fields):
        """
        This method calculate vocab counts in each field
        :param dictionary_fields: dictionary of fields
        :return:
            num_vocab_dict:  dictionary of vocab counts in each field
        """
        num_vocab_dict = dict()
        num_vocab_dict["num_label"] = len(dictionary_fields["label_field"].vocab)
        return num_vocab_dict
