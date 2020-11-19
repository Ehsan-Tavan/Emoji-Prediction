#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable-msg=no-member

"""
evaluation_helper.py is a evaluation file for writing evaluation methods
"""

import torch
from emoji_prediction.config.cnn_config import DEVICE

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "11/19/2020"


def binary_accuracy(preds, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == target).float() # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)   # get the index of the max probability
    correct = max_preds.squeeze(1).eq(target)
    return correct.sum().to(DEVICE) / torch.FloatTensor([target.shape[0]]).to(DEVICE)
