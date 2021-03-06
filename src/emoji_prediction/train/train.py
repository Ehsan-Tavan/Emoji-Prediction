#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
train.py is written for train model
"""

import logging
import random
import itertools
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from emoji_prediction.tools.evaluation_helper import categorical_accuracy, top_n_accuracy
from emoji_prediction.config.cnn_config import DEVICE


__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "12/26/2020"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def lr_scheduler(optimizer, epoch, lr_decay=0.3, lr_decay_epoch=2, number_of_decay=5):
    """
    lr_scheduler method is written for decay learning rate by a factor of lr_decay
    every lr_decay_epoch epochs
    :param optimizer: input optimizer
    :param epoch: epoch number
    :param lr_decay: the rate of reduction, multiplied to learning_rate
    :param lr_decay_epoch: epoch number for decay
    :param number_of_decay: total number of learning_rate reductions
    :return:
        optimizer
    """
    if lr_decay_epoch * number_of_decay < epoch:
        return optimizer

    if (epoch+1) % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group["lr"] *= lr_decay
    return optimizer


def batch_augmentation(text, text_lengths, label, augmentation_class, augmentation_methods):
    """
    batch_augmentation method is written for augment input batch
    :param text: input text
    :param text_lengths: text length
    :param label: label
    :param augmentation_class:  augmentation class
    :param augmentation_methods: augmentation methods dictionary
    :return:
    """
    augmentation_text, augmentation_label = list(), list()
    for txt, length, lbl in zip(text.tolist(), text_lengths.tolist(), label.tolist()):
        augment_sen = augmentation_class.__run__(txt, length, augmentation_methods)
        for sen in augment_sen:
            augmentation_text.append(sen)
            augmentation_label.append(lbl)

    # shuffle list
    z = list(zip(augmentation_text, augmentation_label))
    random.shuffle(z)
    augmentation_text, augmentation_label = zip(*z)

    tensor_augmentation_text = torch.FloatTensor(augmentation_text).long()
    tensor_augmentation_label = torch.FloatTensor(augmentation_label).long()

    augmented_text, augmented_label = tensor_augmentation_text, tensor_augmentation_label
    return augmented_text, augmented_label


def test_augmentation(text, text_lengths, augmentation_class):
    """
    test_augmentation method is written for augment input text in evaluation
    :param text: input text
    :param text_lengths: text length
    :param augmentation_class:  augmentation class
    :return:
    """
    augmentation_text = augmentation_class.test_augment(text, text_lengths)
    augmentation_text.append(text)
    augmentation_text = torch.FloatTensor(augmentation_text).long()
    return augmentation_text


def train(model, iterator, optimizer, criterion, epoch, augmentation_class=None,
          augmentation_methods=None, lr_decay=False, include_length=False):
    """
    train method is written for train model
    :param model: your creation model
    :param iterator: train iterator
    :param optimizer: your optimizer
    :param criterion: your criterion
    :param epoch: training epoch
    :param augmentation_class: augmentation class
    :param augmentation_methods: augmentation method dictionary
    :param lr_decay: if lr_decay is True learning_decay is use
    :param include_length: if true input length is given to the model
    """
    if lr_decay:
        optimizer = lr_scheduler(optimizer, epoch)
        print(f"On epoch {epoch + 1} learning rate is {optimizer.param_groups[0]['lr']}")

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    iter_len = len(iterator)
    n_batch = 0

    # start training model
    for batch in iterator:
        n_batch += 1
        optimizer.zero_grad()
        text, text_lengths = batch.text
        label = batch.label

        # augment input batch
        if augmentation_class is not None:
            text, label = batch_augmentation(text, text_lengths, label,
                                             augmentation_class, augmentation_methods)
            text = text.to(DEVICE)
            label = label.to(DEVICE)

        # predict output
        if include_length:
            predictions = model(text, text_lengths)
        else:
            predictions = model(text)
        # calculate loss
        loss = criterion(predictions, label)

        # calculate accuracy
        acc = categorical_accuracy(predictions, label.to(DEVICE))

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


def evaluate(model, iterator, criterion, include_length=False):
    """
    evaluate method is written for for evaluate model
    :param model: your creation model
    :param iterator: your iterator
    :param criterion: your criterion
    :param include_length: if true input length is given to the model
    :return:
        loss: loss of all  data
        acc: accuracy of all  data
        precision: precision for each class of data
        recall: recall for each class of data
        f-score: F1-score for each class of data
        total_fscore: F1-score of all  data
    """
    # define evaluate_parameters_dict to save output result
    evaluate_parameters_dict = {"loss": 0, "acc": 0, "top_3_acc": 0,
                                "top_5_acc": 0, "precision": 0, "recall": 0,
                                "f-score": 0, "total_fscore": 0}
    total_predict = []
    total_label = []
    # put model in evaluate model
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # predict input data
            text, text_lengths = batch.text

            if include_length:
                predictions = model(text, text_lengths)
            else:
                predictions = model(text)

            total_predict.append(predictions.cpu().numpy())
            total_label.append(batch.label.cpu().numpy())

            # calculate loss
            loss = criterion(predictions, batch.label)

            # calculate accuracy
            acc = categorical_accuracy(predictions, batch.label)
            top_3_acc = top_n_accuracy(predictions.cpu().numpy(), batch.label.cpu().numpy(), n=3)
            top_5_acc = top_n_accuracy(predictions.cpu().numpy(), batch.label.cpu().numpy(), n=5)

            # # save model result
            evaluate_parameters_dict["loss"] += loss.item()
            evaluate_parameters_dict["acc"] += acc.item()
            evaluate_parameters_dict["top_3_acc"] += top_3_acc
            evaluate_parameters_dict["top_5_acc"] += top_5_acc

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

    evaluate_parameters_dict["top_3_acc"] = \
        evaluate_parameters_dict["top_3_acc"] / len(iterator)

    evaluate_parameters_dict["top_5_acc"] = \
        evaluate_parameters_dict["top_5_acc"] / len(iterator)

    return evaluate_parameters_dict


def evaluate_aug_text(model, iterator, include_length=False, augmentation_class=None):
    """
    evaluate_aug_text method is written for text augmentation for test sample
    :param model: your creation model
    :param iterator: your iterator
    :param include_length: if true input length is given to the model
    :param augmentation_class: augmentation class
    :return:
        acc: accuracy of all  data
        precision: precision for each class of data
        recall: recall for each class of data
        f-score: F1-score for each class of data
        total_fscore: F1-score of all  data
    """
    # define evaluate_parameters_dict to save output result
    evaluate_parameters_dict = {"acc": 0, "precision": 0, "recall": 0,
                                "f-score": 0, "total_fscore": 0}
    total_predict = []
    total_label = []

    def most_frequent(pred):
        occurence_count = Counter(pred)
        return occurence_count.most_common(1)[0][0]

    # put model in evaluate model
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # predict input data
            text, text_lengths = batch.text
            label = batch.label

            for sample, length, lbl in zip(text.tolist(), text_lengths.tolist(), label.tolist()):
                augment_sample = test_augmentation(sample, length, augmentation_class).to(DEVICE)
                if include_length:
                    augment_length = [length] * len(augment_sample.tolist())
                    augment_length = torch.FloatTensor(augment_length).long()
                    aug_pred = model(augment_sample, augment_length).tolist()
                else:
                    aug_pred = model(augment_sample).tolist()
                res = [np.argmax(i) for i in aug_pred]
                pred = most_frequent(res)
                total_predict.append(pred)
                total_label.append(lbl)
                # predictions = torch.FloatTensor(predictions).to(DEVICE)

    evaluate_parameters_dict["acc"] = accuracy_score(y_true=total_label, y_pred=total_predict)

    # calculate precision, recall and f_score
    evaluate_parameters_dict["precision"], evaluate_parameters_dict["recall"],\
        evaluate_parameters_dict["f-score"], _ = \
        precision_recall_fscore_support(y_true=total_label,
                                        y_pred=total_predict)

    # calculate total f-score of all data
    evaluate_parameters_dict["total_fscore"] = f1_score(y_true=total_label,
                                                        y_pred=total_predict,
                                                        average="weighted")

    return evaluate_parameters_dict
