#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
train.py is written for train model
"""

import logging
import itertools
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
from emoji_prediction.tools.evaluation_helper import categorical_accuracy

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.3"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "11/03/2020"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def train(model, iterator, optimizer, criterion):
    """
    train method is written for train model
    :param model: your creation model
    :param iterator: train iterator
    :param optimizer: your optimizer
    :param criterion: your criterion
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    iter_len = len(iterator)
    n_batch = 0

    # start training model
    for batch in iterator:
        n_batch += 1
        optimizer.zero_grad()

        # predict output
        predictions = model(batch.text)

        # calculate loss
        loss = criterion(predictions, batch.label)

        # calculate accuracy
        acc = categorical_accuracy(predictions, batch.label)

        # back-propagate loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if (n_batch % (iter_len//5)) == 0:
            logging.info(f"\t train on: {(n_batch / iter_len) * 100:.2f}% of samples")
            logging.info(f"\t accuracy : {(epoch_acc/n_batch) * 100 :.2f}%")
            logging.info(f"\t loss : {(epoch_loss/n_batch):.4f}")
            logging.info("________________________________________________\n")


def evaluate(model, iterator, criterion):
    """
    evaluate method is written for for evaluate model
    :param model: your creation model
    :param iterator: your iterator
    :param criterion: your criterion
    :return:
        loss: loss of all  data
        acc: accuracy of all  data
        precision: precision for each class of data
        recall: recall for each class of data
        f-score: F1-score for each class of data
        total_fscore: F1-score of all  data
    """
    # define evaluate_parameters_dict to save output result
    evaluate_parameters_dict = {"loss": 0, "acc": 0, "precision": 0,
                                "recall": 0, "f-score": 0, "total_fscore": 0}
    total_predict = []
    total_label = []
    # put model in evaluate model
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # predict input data
            predictions = model(batch.text)

            total_predict.append(predictions.cpu().numpy())
            total_label.append(batch.label.cpu().numpy())

            # calculate loss
            loss = criterion(predictions, batch.label)

            # calculate accuracy
            acc = categorical_accuracy(predictions, batch.label)

            # # save model result
            evaluate_parameters_dict["loss"] += loss.item()
            evaluate_parameters_dict["acc"] += acc.item()

    total_predict = list(itertools.chain.from_iterable(total_predict))
    total_predict = list(np.argmax(total_predict, axis=1))
    total_label = list(itertools.chain.from_iterable(total_label))

    # calculate precision, recall and f_score
    evaluate_parameters_dict["precision"], evaluate_parameters_dict["recall"],\
        evaluate_parameters_dict["f-score"], _ = \
        precision_recall_fscore_support(y_true=total_label,
                                        y_pred=total_predict)

    # calculate total f-score of all data
    evaluate_parameters_dict["total_fscore"] = f1_score(y_true=total_label,
                                                        y_pred=total_predict,
                                                        average="weighted")

    evaluate_parameters_dict["loss"] = \
        evaluate_parameters_dict["loss"] / len(iterator)

    evaluate_parameters_dict["acc"] = \
        evaluate_parameters_dict["acc"] / len(iterator)

    return evaluate_parameters_dict
