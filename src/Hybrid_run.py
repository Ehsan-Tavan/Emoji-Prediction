#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error
# pylint: disable-msg=no-member
# pylint: disable-msg=not-callable

"""
Hybrid_run.py is written for run Hybrid model
"""

import time
import logging
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from emoji_prediction.utils.Hybrid_data_util import DataSet, init_weights
from emoji_prediction.methods.Hybrid_model import Hybrid
from emoji_prediction.train.Hybrid_train import train, evaluate
from emoji_prediction.utils.augmentation import Augmentation
from emoji_prediction.tools.log_helper import count_parameters, process_time,\
    model_result_log, model_result_save
from emoji_prediction.config.Hybrid_config import LOG_PATH, TRAIN_NORMAL_DATA_PATH,\
    TEST_NORMAL_DATA_PATH, VALIDATION_NORMAL_DATA_PATH, CHAR_FIELD_PATH, TOKEN_EMB_DIM,\
    LABEL_FIELD_PATH, BATCH_SIZE, DEVICE, MAX_LENGTH, ONE_HOT, CHAR_EMB_DIM, LR_DECAY,\
    BIDIRECTIONAL, LOSS_CURVE_PATH, ACC_CURVE_PATH, N_EPOCHS, TOKEN_FIELD_PATH, N_FILTERS,\
    MODEL_PATH, ADDING_NOISE, SKIPGRAM_NEWS_300D, EMOTION_EMBEDDING_PATH, LSTM_HID_DIM,\
    DROPOUT


__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/11/2020"


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
                           validation_data_path=VALIDATION_NORMAL_DATA_PATH,
                           embedding_path=SKIPGRAM_NEWS_300D,
                           word_emotion_path=EMOTION_EMBEDDING_PATH)

        data_set.load_data(token_field_path=TOKEN_FIELD_PATH,
                           char_field_path=CHAR_FIELD_PATH,
                           label_field_path=LABEL_FIELD_PATH,
                           device=DEVICE, batch_size=BATCH_SIZE,
                           sen_max_len=MAX_LENGTH,
                           one_hot=ONE_HOT)
        return data_set

    @staticmethod
    def init_model(data_set):
        """
        init_model method is written for loading model and
        define loss function and optimizer
        :param data_set:
        :return:
            model: VDCNN model
            criterion: loss function
            optimizer: optimizer function
        """
        # create model
        model = Hybrid(vocab_size=data_set.num_vocab_dict["num_token"],
                       num_char=data_set.num_vocab_dict["num_char"],
                       pad_idx=data_set.pad_idx_dict["token_pad_idx"],
                       char_emd_dim=CHAR_EMB_DIM, one_hot=ONE_HOT,
                       n_class=data_set.num_vocab_dict["num_label"],
                       vocab_emb_dim=TOKEN_EMB_DIM, num_channels=N_FILTERS,
                       lstm_hid_dim=LSTM_HID_DIM, bidirectional=BIDIRECTIONAL,
                       dropout=DROPOUT)

        # initializing model parameters
        model.apply(init_weights)
        logging.info("create model.")

        # copy word embedding vectors to embedding layer
        model.token_embeddings.weight.data.copy_(data_set.embedding_dict["token_embeddings"])
        model.token_embeddings.weight.data[data_set.pad_idx_dict["token_pad_idx"]] =\
            torch.zeros(TOKEN_EMB_DIM)
        model.token_embeddings.weight.requires_grad = True

        if ONE_HOT:
            model.char_embeddings.weight.data.copy_(data_set.embedding_dict["char_embeddings"])
            model.char_embeddings.weight.data[data_set.pad_idx_dict["token_pad_idx"]] = \
                torch.zeros(data_set.num_vocab_dict["num_char"])
            model.char_embeddings.weight.requires_grad = False
        elif not ONE_HOT:
            model.char_embeddings.weight.data[data_set.pad_idx_dict["token_pad_idx"]] = \
                torch.zeros(CHAR_EMB_DIM)
            model.char_embeddings.weight.requires_grad = True

        logging.info(f"The model has {count_parameters(model):,} trainable parameters")

        # define optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        plt.figure()
        plt.plot(kwargs["train_loss"], "r", label="train_loss")
        plt.plot(kwargs["validation_loss"], "b", label="validation_loss")
        plt.plot(kwargs["test_loss"], "g", label="test_loss")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("loss value")
        plt.savefig(LOSS_CURVE_PATH)

        # plot accuracy curves
        plt.figure()
        plt.plot(kwargs["train_acc"], "r", label="train_acc")
        plt.plot(kwargs["validation_acc"], "b", label="validation_acc")
        plt.plot(kwargs["test_acc"], "g", label="test_acc")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("accuracy value")
        plt.savefig(ACC_CURVE_PATH)

    @staticmethod
    def create_augmentation(data_set):
        """
        create_augmentation method is written for create augmentation class
        and define augmentation methods
        :param data_set: data_set class
        :return:
            augmentation_class: augmentation class
            augmentation_methods: augmentation method dictionary

        """
        word2idx = data_set.text_field.vocab.stoi
        idx2word = data_set.text_field.vocab.itos
        vocabs = list(word2idx.keys())
        augmentation_class = Augmentation(word2idx, idx2word, vocabs)

        # augmentation method dictionary
        augmentation_methods = {
            "delete_randomly": True,
            "replace_similar_words": True,
            "swap_token": True
        }
        return augmentation_class, augmentation_methods

    def run(self, adding_noise=False, lr_decay=False, augmentation=False, test_augmentation=False):
        """
        run method is written for running model
        """
        data_set = self.load_data_set()
        model, criterion, optimizer = self.init_model(data_set)

        best_validation_loss = float("inf")
        best_test_f_score = 0.0

        best_val_loss_model = ""
        best_test_f_score_model = ""

        losses_dict = dict()
        acc_dict = dict()
        losses_dict["train_loss"] = []
        losses_dict["validation_loss"] = []
        losses_dict["test_loss"] = []
        acc_dict["train_acc"] = []
        acc_dict["validation_acc"] = []
        acc_dict["test_acc"] = []

        augmentation_class = None
        augmentation_methods = None
        # call augmentation class
        if augmentation:
            augmentation_class, augmentation_methods = self.create_augmentation(data_set)

        # start training model
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            # adding noise to fully connected layers
            if adding_noise:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name.startswith("w_s1") or name.startswith("w_s2"):
                            param.add_(torch.randn(param.size()).to(DEVICE))

            # train model on train data
            if augmentation:
                train(model=model, iterator=data_set.iterator_dict["train_iterator"],
                      optimizer=optimizer, criterion=criterion, epoch=epoch,
                      augmentation_class=augmentation_class, augmentation_methods=augmentation_methods,
                      lr_decay=lr_decay, include_length=False)
            else:
                train(model=model, iterator=data_set.iterator_dict["train_iterator"],
                      optimizer=optimizer, criterion=criterion, epoch=epoch,
                      lr_decay=lr_decay, include_length=False)

            # compute model result on train data
            train_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["train_iterator"],
                                      criterion=criterion, include_length=False)

            losses_dict["train_loss"].append(train_log_dict["loss"])
            acc_dict["train_acc"].append(train_log_dict["acc"])

            # compute model result on validation data
            valid_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["valid_iterator"],
                                      criterion=criterion, include_length=False)

            losses_dict["validation_loss"].append(valid_log_dict["loss"])
            acc_dict["validation_acc"].append(valid_log_dict["acc"])

            # compute model result on test data
            test_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["test_iterator"],
                                     criterion=criterion, include_length=False)

            losses_dict["test_loss"].append(test_log_dict["loss"])
            acc_dict["test_acc"].append(test_log_dict["acc"])

            end_time = time.time()

            # calculate epoch time
            epoch_mins, epoch_secs = process_time(start_time, end_time)

            # save model when loss in validation data is decrease
            if valid_log_dict["loss"] < best_validation_loss:
                best_validation_loss = valid_log_dict["loss"]
                torch.save(model.state_dict(),
                           MODEL_PATH + f"model_epoch{epoch + 1}_loss_"
                           f"{valid_log_dict['loss']}.pt")

            # save model when fscore in test data is increase
            if test_log_dict["total_fscore"] > best_test_f_score:
                best_test_f_score = test_log_dict["total_fscore"]
                torch.save(model.state_dict(),
                           MODEL_PATH + f"model_epoch{epoch + 1}"
                           f"_fscore_{test_log_dict['total_fscore']}.pt")

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
    MYCLASS.run(adding_noise=ADDING_NOISE, lr_decay=LR_DECAY)
