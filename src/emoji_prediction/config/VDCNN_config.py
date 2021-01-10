#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member

"""
VDCNN_config.py is a module for VDCNN model config
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

LOSS_CURVE_PATH = "../models/tmp/Curves/loss_curve.png"
ACC_CURVE_PATH = "../models/tmp/Curves/accuracy_curve.png"

TEXT_FIELD_PATH = "../models/tmp/Fields/text_field"
LABEL_FIELD_PATH = "../models/tmp/Fields/label_field"
LOG_PATH = "../models/tmp/Logs/log.txt"
MODEL_PATH = "../models/tmp/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 10
BATCH_SIZE = 128
EMBEDDING_DIM = 16
N_FILTERS = 256
LINEAR_UNITS = 2048
DROPOUT = 0.5
MAX_LENGTH = 300 # 1014 in original paper
DEPTH = 9

SHORTCUT = False
ONE_HOT = False
ADDING_NOISE = False
LR_DECAY = False
