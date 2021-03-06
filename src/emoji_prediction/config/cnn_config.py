#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member

"""
cnn_config.py is a module for cnn model config
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
__date__ = "01/12/2021"

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

TEXT_FIELD_PATH = "../models/tmp/Fields/text_field"
LABEL_FIELD_PATH = "../models/tmp/Fields/label_field"
LOG_PATH = "../models/tmp/Logs/log.txt"
TEST_AUG_LOG_PATH = "../models/tmp/Logs/test_aug_log.txt"
MODEL_PATH = "../models/tmp/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 20
EMOTION_EMBEDDING_DIM = 10
BATCH_SIZE = 256
EMBEDDING_DIM = 300
FILTER_SIZE = [3, 4, 5]
N_FILTERS = 256
START_DROPOUT = 0.1
MIDDLE_DROPOUT = 0.2
END_DROPOUT = 0.3
MIN_FREQ = 20
MAX_LENGTH = None

ADDING_NOISE = False
LR_DECAY = False
TRAIN_AUGMENTATION = False
TEST_AUGMENTATION = False
USE_EMOTION = True
