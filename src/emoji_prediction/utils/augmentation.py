#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
augmentation.py is writen for augment data in dataLoader
"""

import random
import gensim
import pickle as pkl
from emoji_prediction.config.cnn_config import SKIPGRAM_NEWS_300D

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "11/12/2020"


class Augmentation:
    """
    In this class we implement several augmentation methods on text
    """
    def __init__(self, word2idx_pickle_path, idx2word_pickle_path):
        self.gmodel = gensim.models.KeyedVectors.load_word2vec_format(
            SKIPGRAM_NEWS_300D
        )
        self.word2idx, self.idx2word, vocabs = self.load_word_idx(
            word2idx_pickle_path,
            idx2word_pickle_path)
        self.word_dict = self.find_similar_word_skipgram(vocabs)

    @staticmethod
    def load_word_idx(word2idx_pickle_path, idx2word_pickle_path):
        """
        load_word_idx method is written for load word index
        :param word2idx_pickle_path:
        :param idx2word_pickle_path:
        :return:
            word2idx: word to index dictionary
            idx2word: index to word dictionary
            vocabs: vocabs in word to index
        """
        with open(word2idx_pickle_path, "rb") as f:
            word2idx = pkl.load(f)
        with open(idx2word_pickle_path, "rb") as f:
            idx2word = pkl.load(f)

        vocabs = list(word2idx.keys())
        return word2idx, idx2word, vocabs

    @staticmethod
    def delete_randomly(input_sequence):
        """
        delete_randomly method is written for deleting tokens randomly
        :param input_sequence: sequence(=sentence) of word index
        :return:
            output_sequence: list of sequence(=sentence) of word index
        """
        output_sequence = list()
        # len of input text = number of tokens
        sen_len = len(input_sequence)
        # num_selected_token is number of token that we want to delete
        num_selected_token = int(sen_len * 0.2)

        for del_time in range(num_selected_token):
            input_sequence.pop(random.randrange(len(input_sequence)))

        output_sequence.append(input_sequence)
        return output_sequence

    def find_similar_word_skipgram(self, vocabs):
        """
        find_similar_word_skipgram method is written for find 3 similar words using gensim word2vec
        :param vocabs: list of all vocabs
        :return:
            word_dict: dictionary that have token in key and similar tokens in value
        """
        word_dict = dict()
        for token in vocabs:
            sim_word_list = list()
            try:
                # find three similar words
                similar_token = self.gmodel.similar_by_word(token, 3)
                for word in similar_token:
                    # if similarity value is more than 0.6 we use that token
                    if word[1] > 0.6:
                        sim_word_list.append(word[0])
            except: # Exception as ex:
                pass
                #print(ex)

            if len(sim_word_list) >= 1:
                word_dict[token] = sim_word_list
        return word_dict

    def replace_similar_words(self, input_sequence):
        """
        replace_similar_words method is written for
        :param input_sequence: sequence(=sentence) of word index
        :return:
            output_text_list: list of augmented text
        """
        output_text_list = list()

        # calculate len tokenized_text = num token in sentence
        sen_len = len(input_sequence)

        # calculate number of 25 and 50 percent of words
        len_percent_list = list()
        len_percent_list.append(int(sen_len * 0.25))
        len_percent_list.append(int(sen_len * 0.50))

        for len_percent in len_percent_list:
            if len_percent >= 1:
                # selected_token is randomly selected from tokens to replace
                selected_token = random.sample(input_sequence, len_percent)
            else:
                len_percent = 1
                # selected_token is randomly selected from tokens to replace
                selected_token = random.sample(input_sequence, len_percent)

            new_sequence = list()
            for idx in input_sequence:
                try:
                    if idx in selected_token:
                        token = self.idx2word[idx]
                        sim_words = self.word_dict.get(token)
                        if sim_words is not None:
                            selected_sim_words = random.choice(sim_words)
                            sim_word_idx = self.word2idx[selected_sim_words]
                            idx = sim_word_idx
                except:
                    pass
                new_sequence.append(idx)
            output_text_list.append(new_sequence)
        return output_text_list

    @staticmethod
    def swap_token(input_sequence):
        """
        swap_token method is written for swap words in sentence
        :param input_sequence: sequence(=sentence) of word index
        :return:
            output_sequence: list of sequence(=sentence) of word index
        """
        output_sequence = list()
        random.shuffle(input_sequence)
        output_sequence.append(input_sequence)
        return output_sequence

    def __run__(self, input_sequence, augmentation_methods):
        """
        __run__ method is written for run augmentation method
        :param input_sequence: sequence(=sentence) of word index
        :param augmentation_methods:
        :return:
            final_text_list: list of all augmentation sequence(=sentence)
        """
        final_text_list = list()
        if augmentation_methods["delete_randomly"]:
            delete_seq = self.delete_randomly(input_sequence)
            for text in delete_seq:
                final_text_list.append(text)
        if augmentation_methods["replace_similar_words"]:
            replace_seq = self.replace_similar_words(input_sequence)
            for text in replace_seq:
                final_text_list.append(text)
        if augmentation_methods["swap_token"]:
            swap_seq = self.swap_token(input_sequence)
            for text in swap_seq:
                final_text_list.append(text)
        final_text_list.append(input_sequence)
        return final_text_list

# if __name__ == "__main__":
#     def swap_token(input_sequence):
#         """
#         swap_token method is written for swap words in sentence
#         :param input_sequence: sequence(=sentence) of word index
#         :return:
#             output_sequence: list of sequence(=sentence) of word index
#         """
#         output_sequence = list()
#         random.shuffle(input_sequence)
#         output_sequence.append(input_sequence)
#         return output_sequence
#     x = swap_token([1 , 2, 3 ,4 , 5])
#     print(x)