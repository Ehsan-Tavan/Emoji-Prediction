#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member

"""
cnn_config.py is written for cnn model model
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
__date__ = "11/17/2020"

RAW_NO_MENTION_DATA_PATH = "../data/Raw/first_categori_no_mention_tweets.csv"
RAW_DATA_PATH = "../data/Raw/first_categori_tweets.csv"
TRAIN_NORMAL_NO_MENTION_DATA_PATH = "../data/Processed/" \
                                    "train_first_categori_no_mention_tweets_normal.csv"
TEST_NORMAL_NO_MENTION_DATA_PATH = "../data/Processed/" \
                                   "test_first_categori_no_mention_tweets_normal.csv"
TRAIN_NORMAL_DATA_PATH = "../data/Processed/" \
                                    "train_first_categori_tweets_normal_len5_1015114_final.csv"
TEST_NORMAL_DATA_PATH = "../data/Processed/" \
                                   "test_first_categori_tweets_normal_len5_125323_final.csv"
VALIDATION_NORMAL_DATA_PATH = "../data/Processed/" \
                                   "validation_first_categori_tweets_normal_len5_112791_final.csv"

GLOVE_NEWS_300D = "../data/Embeddings/news_glove_300d_e10.txt"
SKIPGRAM_NEWS_300D = "../data/Embeddings/skipgram_news_300d_30e.txt"
CBOW_NEWS_300D = "../data/Embeddings/cbow_news_300d_30e.txt"

LOSS_CURVE_PATH = "../models/ID_5/Curves/loss_curve.png"
ACC_CURVE_PATH = "../models/ID_5/Curves/accuracy_curve.png"

TEXT_FIELD_PATH = "../models/ID_5/Fields/text_field"
LABEL_FIELD_PATH = "../models/ID_5/Fields/label_field"
LOG_PATH = "../models/ID_5/Logs/log.txt"
MODEL_PATH = "../models/ID_5/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 30
BATCH_SIZE = 100
EMBEDDING_DIM = 300
FILTER_SIZE = [3, 4, 5]
N_FILTERS = 256
START_DROPOUT = 0.1
MIDDLE_DROPOUT = 0.2
END_DROPOUT = 0.3
