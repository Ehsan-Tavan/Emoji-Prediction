#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
# pylint: disable-msg=import-error

"""
bert_model.py is written for bert model
"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "01/02/2020"


class BERTEmoji(nn.Module):
    """
    In this class we implement Bert model for sentiment
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.bert, embedding_dim = self.bert_loader(kwargs["model_config"])

        self.fully_connected_layers = nn.Linear(
            in_features=kwargs["sen_len"]*768, out_features=kwargs["output_size"]
        )

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])

    @staticmethod
    def bert_loader(model_config, dl=False, save=True):
        """
        bert_loader method is written for load bert model
        :param model_config: model_config
        :param dl: if true download preTrain model else load from local
        :param save: if true save preTrain model
        :return:
            model: bert model
            embedding_dim: hidden_size of bert
        """
        if dl:
            # download model
            config = AutoConfig.from_pretrained(model_config["url_path"])
            model = AutoModel.from_pretrained(model_config["url_path"], from_tf=True)

            if save:
                # save model
                model.save_pretrained(model_config["model_path"])
                config.save_pretrained(model_config["model_path"])
        else:
            # load from local
            config = AutoConfig.from_pretrained(model_config["model_path"])
            model = AutoModel.from_pretrained(model_config["model_path"])

        embedding_dim = model.config.to_dict()["hidden_size"]
        return model, embedding_dim

    def forward(self, text):
        # text.size() = [batch_size, sent_len]

        with torch.no_grad():
          embedded = self.start_dropout(self.bert(text)[0])
        # embedded.size() = [batch_size, seq_len, emb_dim]

        embedded = torch.flatten(embedded, start_dim=1)

        predictions = self.fully_connected_layers(embedded)
        # predictions = [batch_size, output_dim]

        return predictions
