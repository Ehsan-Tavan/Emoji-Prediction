#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member

"""
Hybrid_config.py is a module for Hybrid model config
"""

import torch

__author__ = "Ehsan Tavan"
__project__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/10/2020"

RAW_NO_MENTION_DATA_PATH = "../data/Raw/first_categori_no_mention_tweets.csv"
RAW_DATA_PATH = "../data/Raw/first_categori_tweets.csv"
TRAIN_NORMAL_NO_MENTION_DATA_PATH = "../data/Processed/" \
                                    "train_first_categori_no_mention_tweets_normal.csv"
TEST_NORMAL_NO_MENTION_DATA_PATH = "../data/Processed/" \
                                   "test_first_categori_no_mention_tweets_normal.csv"
TRAIN_NORMAL_DATA_PATH = "../data/Processed/" \
                                    "train_first_categori_tweets_normal_5.csv"
TEST_NORMAL_DATA_PATH = "../data/Processed/" \
                                   "train_first_categori_tweets_normal_5.csv"
VALIDATION_NORMAL_DATA_PATH = "../data/Processed/" \
                                   "train_first_categori_tweets_normal_5.csv"

GLOVE_NEWS_300D = "../data/Embeddings/news_glove_300d_e10.txt"
SKIPGRAM_NEWS_300D = "../data/Embeddings/skipgram_news_300d_30e.txt"
CBOW_NEWS_300D = "../data/Embeddings/cbow_news_300d_30e.txt"
EMOTION_EMBEDDING_PATH = "../data/Embeddings/word_emotion_dict.pkl"

LOSS_CURVE_PATH = "../models/tmp/Curves/loss_curve.png"
ACC_CURVE_PATH = "../models/tmp/Curves/accuracy_curve.png"

TOKEN_FIELD_PATH = "../models/tmp/Fields/token_field"
CHAR_FIELD_PATH = "../models/tmp/Fields/char_field"
LABEL_FIELD_PATH = "../models/tmp/Fields/label_field"
LOG_PATH = "../models/tmp/Logs/log.txt"
MODEL_PATH = "../models/tmp/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 10
BATCH_SIZE = 128
CHAR_EMB_DIM = 16
TOKEN_EMB_DIM = 300
N_FILTERS = 512
LSTM_HID_DIM = 128
DROPOUT = 0.2
BIDIRECTIONAL = True
MAX_LENGTH = None

ONE_HOT = False
ADDING_NOISE = False
LR_DECAY = False
