#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error
# pylint: disable-msg=no-member
# pylint: disable-msg=not-callable

"""
cnn_run.py is written for run cnn model
"""

import time
import logging
from collections import OrderedDict
import torch
from torch import optim
from torch import nn
from emoji_prediction.utils.util import DataSet, init_weights
from emoji_prediction.methods.cnn_model import CNN
from emoji_prediction.train.train import train, evaluate
from emoji_prediction.tools.log_helper import count_parameters, process_time,\
    model_result_log, model_result_save
from emoji_prediction.config.cnn_config import LOG_PATH, TRAIN_NORMAL_DATA_PATH,\
    TEST_NORMAL_DATA_PATH, SKIPGRAM_NEWS_300D, EMBEDDING_DIM, DEVICE, N_EPOCHS,\
    MODEL_PATH, N_FILTERS, FILTER_SIZE, START_DROPOUT, MIDDLE_DROPOUT, END_DROPOUT

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "10/20/2020"


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# open log file
LOG_FILE = open(LOG_PATH, "w")

# load data from input file
DATA_SET = DataSet(train_data_path=TRAIN_NORMAL_DATA_PATH,
                   test_data_path=TEST_NORMAL_DATA_PATH,
                   embedding_path=SKIPGRAM_NEWS_300D)
DATA_SET.load_data()

# create model
MODEL = CNN(vocab_size=DATA_SET.num_vocab_dict["num_token"],
            embedding_dim=EMBEDDING_DIM,
            pad_idx=DATA_SET.pad_idx_dict["token_pad_idx"],
            n_filters=N_FILTERS, filter_sizes=FILTER_SIZE,
            output_size=DATA_SET.num_vocab_dict["num_label"],
            start_dropout=START_DROPOUT, middle_dropout=MIDDLE_DROPOUT,
            end_dropout=END_DROPOUT)

# initializing model parameters
MODEL.apply(init_weights)
logging.info("create model.")

# copy word embedding vectors to embedding layer
MODEL.embeddings.weight.data.copy_(DATA_SET.embedding_dict["vocab_embedding_vectors"])
MODEL.embeddings.weight.data[DATA_SET.pad_idx_dict["token_pad_idx"]] = torch.zeros(EMBEDDING_DIM)
MODEL.embeddings.weight.requires_grad = True

logging.info(f"The model has {count_parameters(MODEL):,} trainable parameters")

# define optimizer
OPTIMIZER = optim.Adam(MODEL.parameters())
# define loss function
CRITERION = nn.CrossEntropyLoss(weight=torch.tensor(DATA_SET.class_weight))

# load model into GPU
MODEL = MODEL.to(DEVICE)
CRITERION = CRITERION.to(DEVICE)

BEST_VALID_LOSS = float("inf")
BEST_TEST_FSCORE = 0.0

# start training model
for epoch in range(N_EPOCHS):

    start_time = time.time()

    # train model on train data
    train_log_dict = OrderedDict()
    train_log_dict["train_loss"], train_log_dict["train_acc"] =\
        train(MODEL, DATA_SET.iterator_dict["train_iterator"], OPTIMIZER, CRITERION)

    # compute model result on train data
    _, _, train_log_dict["train_precision"], train_log_dict["train_recall"],\
        train_log_dict["train_fscore"], train_log_dict["train_total_fscore"] =\
        evaluate(MODEL, DATA_SET.iterator_dict["train_iterator_eval"], CRITERION)

    # compute model result on validation data
    valid_log_dict = OrderedDict()
    valid_log_dict["valid_loss"], valid_log_dict["valid_acc"],\
        valid_log_dict["valid_precision"], valid_log_dict["valid_recall"],\
        valid_log_dict["valid_fscore"], valid_log_dict["valid_total_fscore"] =\
        evaluate(MODEL, DATA_SET.iterator_dict["valid_iterator"], CRITERION)

    # compute model result on test data
    test_log_dict = OrderedDict()
    test_log_dict["test_loss"], test_log_dict["test_acc"], \
        test_log_dict["test_precision"], test_log_dict["test_recall"], \
        test_log_dict["test_fscore"], test_log_dict["test_total_fscore"] = \
        evaluate(MODEL, DATA_SET.iterator_dict["test_iterator"], CRITERION)

    end_time = time.time()

    # calculate epoch time
    epoch_mins, epoch_secs = process_time(start_time, end_time)

    # save model when loss in validation data is decrease
    if valid_log_dict["valid_loss"] < BEST_VALID_LOSS:
        BEST_VALID_LOSS = valid_log_dict["valid_loss"]
        torch.save(MODEL.state_dict(),
                   MODEL_PATH + f"model_epoch{epoch + 1}_loss_{valid_log_dict['valid_loss']}.pt")

    # save model when fscore in class 2 of test data is increase
    if test_log_dict["test_total_fscore"] > BEST_TEST_FSCORE:
        BEST_TEST_FSCORE = test_log_dict["test_total_fscore"]
        torch.save(MODEL.state_dict(),
                   MODEL_PATH + f"model_epoch{epoch + 1}"
                   f"_fscore_{test_log_dict['test_total_fscore']}.pt")

    # show model result
    logging.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    model_result_log(train_log_dict, valid_log_dict, test_log_dict)

    # save model result in log file
    LOG_FILE.write(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n")
    model_result_save(LOG_FILE, train_log_dict, valid_log_dict, test_log_dict)

# save final model
torch.save(MODEL.state_dict(), MODEL_PATH + "final_model.pt")
