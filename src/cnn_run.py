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
import matplotlib.pyplot as plt
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
    MODEL_PATH, N_FILTERS, FILTER_SIZE, START_DROPOUT, MIDDLE_DROPOUT, END_DROPOUT,\
    LOSS_CURVE_PATH, ACC_CURVE_PATH

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.1.1"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "10/30/2020"


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class RunModel:
    """
    In this class we start training and testing model
    """
    def __init__(self):
        # open log file
        self.log_file = open(LOG_PATH, "w")

    @staticmethod
    def load_data_set():
        """
        load_data_set method is written for load input data and iterators
        :return:
            data_set: data_set
        """
        # load data from input file
        data_set = DataSet(train_data_path=TRAIN_NORMAL_DATA_PATH,
                           test_data_path=TEST_NORMAL_DATA_PATH,
                           embedding_path=SKIPGRAM_NEWS_300D)
        data_set.load_data()
        return data_set

    @staticmethod
    def init_model(data_set):
        """
        init_model method is written for loading model and
        define loss function and optimizer
        :param data_set:
        :return:
            model: deepmoji model
            criterion: loss function
            optimizer: optimizer function
        """
        # create model
        model = CNN(vocab_size=data_set.num_vocab_dict["num_token"],
                    embedding_dim=EMBEDDING_DIM,
                    pad_idx=data_set.pad_idx_dict["token_pad_idx"],
                    n_filters=N_FILTERS, filter_sizes=FILTER_SIZE,
                    output_size=data_set.num_vocab_dict["num_label"],
                    start_dropout=START_DROPOUT, middle_dropout=MIDDLE_DROPOUT,
                    end_dropout=END_DROPOUT)

        # initializing model parameters
        model.apply(init_weights)
        logging.info("create model.")

        # copy word embedding vectors to embedding layer
        model.embeddings.weight.data.copy_(data_set.embedding_dict["vocab_embedding_vectors"])
        model.embeddings.weight.data[data_set.pad_idx_dict["token_pad_idx"]] = \
            torch.zeros(EMBEDDING_DIM)
        model.embeddings.weight.requires_grad = True

        logging.info(f"The model has {count_parameters(model):,} trainable parameters")

        # define optimizer
        optimizer = optim.Adam(model.parameters())
        # define loss function
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(data_set.class_weight))

        # load model into GPU
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        return model, criterion, optimizer

    @staticmethod
    def draw_curves(**kwargs):
        """
        draw_curves method is written for drawing loss and accuracy curve
        """
        # plot loss curves
        plt.plot(kwargs["train_loss"], "r", label="train_loss")
        plt.plot(kwargs["validation_loss"], "b", label="validation_loss")
        plt.plot(kwargs["test_loss"], "g", label="test_loss")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("loss value")
        plt.savefig(LOSS_CURVE_PATH)

        # clear figure command
        plt.clf()

        # plot accuracy curves
        plt.plot(kwargs["train_acc"], "r", label="train_acc")
        plt.plot(kwargs["validation_acc"], "b", label="validation_acc")
        plt.plot(kwargs["test_acc"], "g", label="test_acc")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("accuracy value")
        plt.savefig(ACC_CURVE_PATH)

    def run(self):
        """
        run method is written for running model
        """
        data_set = self.load_data_set()
        model, criterion, optimizer = self.init_model(data_set)

        best_validation_loss = float("inf")
        best_test_f_score = 0.0

        losses_dict = OrderedDict()
        acc_dict = OrderedDict()
        losses_dict["train_loss"] = []
        losses_dict["validation_loss"] = []
        losses_dict["test_loss"] = []
        acc_dict["train_acc"] = []
        acc_dict["validation_acc"] = []
        acc_dict["test_acc"] = []

        # start training model
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            # train model on train data
            train_log_dict = OrderedDict()
            train_log_dict["train_loss"], train_log_dict["train_acc"] =\
                train(model, data_set.iterator_dict["train_iterator"], optimizer, criterion)
            losses_dict["train_loss"].append(train_log_dict["train_loss"])
            acc_dict["train_acc"].append(train_log_dict["train_acc"])

            # compute model result on train data
            _, _, train_log_dict["train_precision"], train_log_dict["train_recall"],\
                train_log_dict["train_fscore"], train_log_dict["train_total_fscore"] =\
                evaluate(model, data_set.iterator_dict["train_iterator_eval"], criterion)

            # compute model result on validation data
            valid_log_dict = OrderedDict()
            valid_log_dict["valid_loss"], valid_log_dict["valid_acc"],\
                valid_log_dict["valid_precision"], valid_log_dict["valid_recall"],\
                valid_log_dict["valid_fscore"], valid_log_dict["valid_total_fscore"] =\
                evaluate(model, data_set.iterator_dict["valid_iterator"], criterion)

            losses_dict["validation_loss"].append(valid_log_dict["valid_loss"])
            acc_dict["validation_acc"].append(valid_log_dict["valid_acc"])

            # compute model result on test data
            test_log_dict = OrderedDict()
            test_log_dict["test_loss"], test_log_dict["test_acc"], \
                test_log_dict["test_precision"], test_log_dict["test_recall"], \
                test_log_dict["test_fscore"], test_log_dict["test_total_fscore"] = \
                evaluate(model, data_set.iterator_dict["test_iterator"], criterion)

            losses_dict["test_loss"].append(test_log_dict["test_loss"])
            acc_dict["test_acc"].append(test_log_dict["test_acc"])

            end_time = time.time()

            # calculate epoch time
            epoch_mins, epoch_secs = process_time(start_time, end_time)

            # save model when loss in validation data is decrease
            if valid_log_dict["valid_loss"] < best_validation_loss:
                best_validation_loss = valid_log_dict["valid_loss"]
                torch.save(model.state_dict(),
                           MODEL_PATH + f"model_epoch{epoch + 1}_loss_"
                           f"{valid_log_dict['valid_loss']}.pt")

            # save model when fscore in class 2 of test data is increase
            if test_log_dict["test_total_fscore"] > best_test_f_score:
                best_test_f_score = test_log_dict["test_total_fscore"]
                torch.save(model.state_dict(),
                           MODEL_PATH + f"model_epoch{epoch + 1}"
                           f"_fscore_{test_log_dict['test_total_fscore']}.pt")

            # show model result
            logging.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            model_result_log(train_log_dict, valid_log_dict, test_log_dict)

            # save model result in log file
            self.log_file.write(f"Epoch: {epoch + 1:02} | Epoch Time: "
                                f"{epoch_mins}m {epoch_secs}s\n")
            model_result_save(self.log_file, train_log_dict, valid_log_dict, test_log_dict)

        # save final model
        torch.save(model.state_dict(), MODEL_PATH + "final_model.pt")

        # plot curve
        self.draw_curves(train_acc=acc_dict["train_acc"], validation_acc=acc_dict["validation_acc"],
                         test_acc=acc_dict["test_acc"], train_loss=losses_dict["train_loss"],
                         validation_loss=losses_dict["validation_loss"],
                         test_loss=losses_dict["test_loss"])


if __name__ == "__main__":
    MYCLASS = RunModel()
    MYCLASS.run()
