#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error
# pylint: disable-msg=no-member
# pylint: disable-msg=not-callable

"""
bert_run.py is written for run bert model
"""

import time
import logging
from torch import optim
from torch import nn
import torch
import matplotlib.pyplot as plt
from emoji_prediction.methods.bert_model import BERTEmoji
from emoji_prediction.utils.bert_data_util import DataSet
from emoji_prediction.tools.log_helper import count_parameters
from emoji_prediction.train.train import train, evaluate, evaluate_aug_text
from emoji_prediction.utils.augmentation import Augmentation
from emoji_prediction.tools.log_helper import process_time, model_result_log, model_result_save,\
    test_aug_result_log, test_aug_result_save
from emoji_prediction.config.bert_config import MODEL_NAME, TRAIN_NORMAL_DATA_PATH, TEST_NORMAL_DATA_PATH,\
    VALIDATION_NORMAL_DATA_PATH, START_DROPOUT, FINAL_DROPOUT, BERT_CONFIG, PARSBERT_CONFIG, ALBERT_CONFIG,\
    DEVICE, N_EPOCHS, SEN_LEN, LR_DECAY, TRAIN_AUGMENTATION, TEST_AUGMENTATION

__author__ = "Ehsan Tavan"
__project__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/03/2020"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class RunModel:
    """
    In this class we start training and testing model
    """
    def __init__(self):
        self.log_file = None
        self.model_name = MODEL_NAME

    @staticmethod
    def load_data_set(model_config):
        """
        load_data_set method is written for load input data and iterators
        :param model_config
        :return:
            data_set: data_set
        """
        # load data from input file
        data_set = DataSet(train_data_path=TRAIN_NORMAL_DATA_PATH,
                           validation_data_path=VALIDATION_NORMAL_DATA_PATH,
                           test_data_path=TEST_NORMAL_DATA_PATH)
        data_set.load_data(model_config)
        return data_set

    @staticmethod
    def init_model(data_set, model_config):
        """
        init_model method is written for loading model and
        define loss function and optimizer
        :param data_set:
        :param model_config:
        :return:
            model: parsbert model
            criterion: loss function
            optimizer: optimizer function
        """
        # create model
        model = BERTEmoji(output_size=data_set.num_vocab_dict["num_label"],
                          start_dropout=START_DROPOUT, final_dropout=FINAL_DROPOUT,
                          model_config=model_config, sen_len=SEN_LEN)

        logging.info("create model.")

        # freeze bert model parameters
        for name, param in model.named_parameters():
            if name.startswith("bert"):
                param.requires_grad = False

        logging.info(f"The model has {count_parameters(model):,} trainable parameters")

        # define optimizer
        optimizer = optim.Adam(model.parameters())
        # define loss function
        criterion = nn.CrossEntropyLoss()

        # load model into GPU
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        return model, criterion, optimizer

    @staticmethod
    def draw_curves(model_config, **kwargs, ):
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
        plt.savefig(model_config["loc_curves_path"])

        # clear figure command
        plt.clf()

        # plot accuracy curves
        plt.figure()
        plt.plot(kwargs["train_acc"], "r", label="train_acc")
        plt.plot(kwargs["validation_acc"], "b", label="validation_acc")
        plt.plot(kwargs["test_acc"], "g", label="test_acc")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("accuracy value")
        plt.savefig(model_config["acc_curves_path"])

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
        word2idx = data_set.word2idx
        idx2word = data_set.idx2word
        vocabs = data_set.vocabs

        augmentation_class = Augmentation(word2idx, idx2word, vocabs)

        # augmentation method dictionary
        augmentation_methods = {
            "delete_randomly": True,
            "replace_similar_words": True,
            "swap_token": True
        }
        return augmentation_class, augmentation_methods

    def run(self, model_name, lr_decay=False, augmentation=False, test_augmentation=False):
        """
        run method is written for running model
        """
        # select model
        model_config = dict()
        if model_name == "bert":
            model_config = BERT_CONFIG
        elif model_name == "parsbert":
            model_config = PARSBERT_CONFIG
        elif model_name == "albert":
            model_config = ALBERT_CONFIG

        # open log file
        self.log_file = open(model_config["log_path"], "w")

        # load data_set iterators
        data_set = self.load_data_set(model_config)
        # create model
        model, criterion, optimizer = self.init_model(data_set, model_config)

        best_validation_loss = float("inf")
        best_test_f_score = 0.0

        best_val_loss_model = ""
        best_test_f_score_model = ""

        losses_dict = dict()
        acc_dict = dict()
        losses_dict["train_loss"] = []
        losses_dict["dev_loss"] = []
        losses_dict["test_loss"] = []
        acc_dict["train_acc"] = []
        acc_dict["dev_acc"] = []
        acc_dict["test_acc"] = []

        augmentation_class = None
        augmentation_methods = None

        # call augmentation class
        if augmentation:
            augmentation_class, augmentation_methods = self.create_augmentation(data_set)

        # start training model
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            # train model on train data
            if augmentation:
                train(model=model, iterator=data_set.iterator_dict["train_iterator"],
                      optimizer=optimizer, criterion=criterion, epoch=epoch,
                      augmentation_class=augmentation_class, augmentation_methods=augmentation_methods,
                      lr_decay=lr_decay)
            else:
                train(model=model, iterator=data_set.iterator_dict["train_iterator"],
                      optimizer=optimizer, criterion=criterion, epoch=epoch,
                      lr_decay=lr_decay)

            # compute model result on train data
            train_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["train_iterator"],
                                      criterion=criterion)

            losses_dict["train_loss"].append(train_log_dict["loss"])
            acc_dict["train_acc"].append(train_log_dict["acc"])

            # compute model result on dev data
            valid_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["valid_iterator"],
                                      criterion=criterion)

            losses_dict["dev_loss"].append(valid_log_dict["loss"])
            acc_dict["dev_acc"].append(valid_log_dict["acc"])

            # compute model result on test data
            test_log_dict = evaluate(model=model, iterator=data_set.iterator_dict["test_iterator"],
                                     criterion=criterion)

            losses_dict["test_loss"].append(test_log_dict["loss"])
            acc_dict["test_acc"].append(test_log_dict["acc"])

            end_time = time.time()

            # calculate epoch time
            epoch_mins, epoch_secs = process_time(start_time, end_time)

            # save model when loss in validation data is decrease
            if valid_log_dict["loss"] < best_validation_loss:
                best_validation_loss = valid_log_dict["loss"]
                torch.save(model.state_dict(),
                           model_config["save_model_path"] + f"model_epoch{epoch + 1}_loss_"
                           f"{valid_log_dict['loss']}.pt")
                best_val_loss_model = f"model_epoch{epoch + 1}_loss_" \
                    f"{valid_log_dict['loss']}.pt"

            # save model when fscore of test data is increase
            if test_log_dict["total_fscore"] > best_test_f_score:
                best_test_f_score = test_log_dict["total_fscore"]
                torch.save(model.state_dict(),
                           model_config["save_model_path"] + f"model_epoch{epoch + 1}"
                           f"_fscore_{test_log_dict['total_fscore']}.pt")
                best_test_f_score_model = f"model_epoch{epoch + 1}" \
                    f"_fscore_{test_log_dict['total_fscore']}.pt"

            # show model result
            logging.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            model_result_log(train_log_dict, test_log_dict, test_log_dict)

            # save model result in log file
            self.log_file.write(f"Epoch: {epoch + 1:02} | Epoch Time: "
                                f"{epoch_mins}m {epoch_secs}s\n")
            model_result_save(self.log_file, train_log_dict, test_log_dict, test_log_dict)

        # save final model
        torch.save(model.state_dict(), model_config["save_model_path"] + "final_model.pt")

        # test augmentation
        if test_augmentation:
            self.eval_test_augmentation(best_val_loss_model=best_val_loss_model,
                                        best_test_f_score_model=best_test_f_score_model,
                                        data_set=data_set, aug_class=augmentation_class,
                                        model_config=model_config)

        # plot curve
        self.draw_curves(train_acc=acc_dict["train_acc"], validation_acc=acc_dict["dev_acc"],
                         test_acc=acc_dict["test_acc"], train_loss=losses_dict["train_loss"],
                         validation_loss=losses_dict["dev_loss"],
                         test_loss=losses_dict["test_loss"], model_config=model_config)

    def eval_test_augmentation(self, best_val_loss_model, best_test_f_score_model,
                               data_set, aug_class, model_config):
        """
        eval_test_augmentation method is written for test augmentation
        :param best_val_loss_model: model path
        :param best_test_f_score_model: model path
        :param data_set: data_set class
        :param aug_class: augmentation class
        :param model_config:
        """
        log_file = open(model_config["test_aug_log_path"], "w")
        model, _, _ = self.init_model(data_set, model_config)
        model.load_state_dict(torch.load(model_config["save_model_path"] + best_val_loss_model,
                                         map_location=DEVICE))
        evaluate_parameters_dict = evaluate_aug_text(model=model, include_length=True,
                                                     iterator=data_set.iterator_dict["test_iterator"],
                                                     augmentation_class=aug_class)

        test_aug_result_log(evaluate_parameters_dict)
        log_file.write("*****************best_val_loss_model*****************\n")
        log_file.flush()
        test_aug_result_save(log_file, evaluate_parameters_dict)

        model, _, _ = self.init_model(data_set, model_config)
        model.load_state_dict(torch.load(model_config["save_model_path"] + best_test_f_score_model,
                                         map_location=DEVICE))
        evaluate_parameters_dict = evaluate_aug_text(model=model, include_length=True,
                                                     iterator=data_set.iterator_dict["test_iterator"],
                                                     augmentation_class=aug_class)
        test_aug_result_log(evaluate_parameters_dict)
        log_file.write("*****************best_test_f_score_model*****************\n")
        log_file.flush()
        test_aug_result_save(log_file, evaluate_parameters_dict)


if __name__ == "__main__":
    MYCLASS = RunModel()
    MYCLASS.run(MODEL_NAME, lr_decay=LR_DECAY,augmentation=TRAIN_AUGMENTATION,
                test_augmentation=TEST_AUGMENTATION)
