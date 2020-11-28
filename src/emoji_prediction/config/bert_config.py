#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member

"""
bert_config.py is written for bert model
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
__date__ = "11/28/2020"

# select from (bert, parsbert, albert)
MODEL_NAME = "pasrbert"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 50
START_DROPOUT = 0.1
MIDDLE_DROPOUT = 0.2
FINAL_DROPOUT = 0.3
BATCH_SIZE = 64
SEN_LEN = 35

TRAIN_NORMAL_DATA_PATH = "../data/Processed/" \
                                    "train_first_categori_tweets_normal_5.csv"
TEST_NORMAL_DATA_PATH = "../data/Processed/" \
                                   "train_first_categori_tweets_normal_5.csv"
VALIDATION_NORMAL_DATA_PATH = "../data/Processed/" \
                                   "train_first_categori_tweets_normal_5.csv"

BERT_CONFIG = {"url_path": "HooshvareLab/bert-fa-base-uncased",
               "tokenizer_path": "../models/bert/tokenizer/",
               "model_path": "../models/bert/model/",
               "log_path": "../models/bert/Logs/log.txt",
               "text_field_path": "../models/bert/Fields/text_field.Field",
               "label_field_path": "../models/bert/Fields/label_field.Field",
               "loc_curves_path": "../models/bert/Curves/loss_curve.png",
               "acc_curves_path": "../models/bert/Curves/accuracy_curve.png",
               "save_model_path": "../models/bert/"}

PARSBERT_CONFIG = {"url_path": "HooshvareLab/bert-fa-base-uncased",
                   "tokenizer_path": "../models/parsbert/tokenizer/",
                   "model_path": "../models/parsbert/model/",
                   "log_path": "../models/parsbert/Logs/log.txt",
                   "text_field_path": "../models/parsbert/Fields/text_field.Field",
                   "label_field_path": "../models/parsbert/Fields/label_field.Field",
                   "loc_curves_path": "../models/parsbert/Curves/loss_curve.png",
                   "acc_curves_path": "../models/parsbert/Curves/accuracy_curve.png",
                   "save_model_path": "../models/parsbert/"}

ALBERT_CONFIG = {"url_path": "m3hrdadfi/albert-fa-base-v2",
                 "tokenizer_path": "../models/albert/tokenizer/",
                 "model_path": "../models/albert/model/",
                 "log_path": "../models/albert/Logs/log.txt",
                 "text_field_path": "../models/albert/Fields/text_field.Field",
                 "label_field_path": "../models/albert/Fields/label_field.Field",
                 "loc_curves_path": "../models/albert/Curves/loss_curve.png",
                 "acc_curves_path": "../models/albert/Curves/accuracy_curve.png",
                 "save_model_path": "../models/albert/"}
