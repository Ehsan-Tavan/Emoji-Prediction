#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
augmentation.py is writen for augment data in dataLoader
"""

import random
import gensim
from emoji_prediction.config.cnn_config import SKIPGRAM_NEWS_300D

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "11/25/2020"


class Augmentation:
    """
    In this class we implement several augmentation methods on text
    """
    def __init__(self, word2idx, idx2word, vocabs):
        self.gmodel = gensim.models.KeyedVectors.load_word2vec_format(
            SKIPGRAM_NEWS_300D
        )

        self.word2idx, self.idx2word = word2idx, idx2word
        self.word_dict = self.find_similar_word_skipgram(vocabs)

    @staticmethod
    def delete_randomly(input_sequence, input_length):
        """
        delete_randomly method is written for deleting tokens randomly
        :param input_sequence: sequence(=sentence) of word index
        :param input_length: input sequence(=sentence) length
        :return:
            output_sequence: list of sequence(=sentence) of word index
        """
        output_sequence = list()
        # len of input text = number of tokens
        input_sequence_tokens = input_sequence[:input_length]
        sen_len = len(input_sequence_tokens)
        # num_selected_token is number of token that we want to delete
        num_selected_token = int(sen_len * 0.2)

        for del_time in range(num_selected_token):
            input_sequence_tokens.pop(random.randrange(len(input_sequence_tokens)))

        for i in range(num_selected_token):
            input_sequence_tokens.append(1)
        final_sequence = input_sequence_tokens + input_sequence[input_length:]
        output_sequence.append(final_sequence)
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

    def replace_similar_words(self, input_sequence, input_length):
        """
        replace_similar_words method is written for
        :param input_sequence: sequence(=sentence) of word index
        :param input_length: input sequence(=sentence) length
        :return:
            output_text_list: list of augmented text
        """
        output_text_list = list()

        # calculate len tokenized_text = num token in sentence
        input_sequence_tokens = input_sequence[:input_length]
        sen_len = len(input_sequence_tokens)

        # calculate number of 25 and 50 percent of words
        len_percent_list = list()
        len_percent_list.append(int(sen_len * 0.25))
        len_percent_list.append(int(sen_len * 0.50))

        for len_percent in len_percent_list:
            if len_percent >= 1:
                # selected_token is randomly selected from tokens to replace
                selected_token = random.sample(input_sequence_tokens, len_percent)
            else:
                len_percent = 1
                # selected_token is randomly selected from tokens to replace
                selected_token = random.sample(input_sequence_tokens, len_percent)

            new_sequence = list()
            for idx in input_sequence_tokens:
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
            new_sequence = new_sequence + input_sequence[input_length:]
            output_text_list.append(new_sequence)
        return output_text_list

    @staticmethod
    def swap_token(input_sequence, input_length):
        """
        swap_token method is written for swap words in sentence
        :param input_sequence: sequence(=sentence) of word index
        :param input_length: input sequence(=sentence) length
        :return:
            output_sequence: list of sequence(=sentence) of word index
        """
        output_sequence = list()
        input_sequence_tokens = input_sequence[:input_length]
        random.shuffle(input_sequence_tokens)
        final_sequencce = input_sequence_tokens + input_sequence[input_length:]
        output_sequence.append(final_sequencce)
        return output_sequence

    @staticmethod
    def split_text(input_sequence, input_length):
        """
        split_text method is written for split sentence
        :param input_sequence: sequence(=sentence) of word index
        :param input_length: input sequence(=sentence) length
        :return:
            output_sequence: list of sequence(=sentence) of word index
        """
        output_sequence = list()
        inter_sequence = list()
        input_sequence_tokens = input_sequence[:input_length]

        # split fifty
        inter_sequence.append(input_sequence_tokens[:int(0.5 * input_length)])
        inter_sequence.append(input_sequence_tokens[int(0.5 * input_length):])
        inter_sequence.append(input_sequence_tokens[int(0.25 * input_length): -int(0.25 * input_length)])

        for seq in inter_sequence:
            for i in range(input_length-(len(seq))):
                seq.append(1)
            output_sequence.append(seq + input_sequence[input_length:])

        return output_sequence

    def __run__(self, input_sequence, input_length, augmentation_methods):
        """
        __run__ method is written for run augmentation method
        :param input_sequence: sequence(=sentence) of word index
        :param input_length: input sequence(=sentence) length
        :param augmentation_methods: augmentation method dictionary
        :return:
            final_text_list: list of all augmented sequence(=sentence)
        """
        final_text_list = list()
        if augmentation_methods["delete_randomly"]:
            delete_seq = self.delete_randomly(input_sequence, input_length)
            for text in delete_seq:
                final_text_list.append(text)
        if augmentation_methods["replace_similar_words"]:
            replace_seq = self.replace_similar_words(input_sequence, input_length)
            for text in replace_seq:
                final_text_list.append(text)
        if augmentation_methods["swap_token"]:
            swap_seq = self.swap_token(input_sequence, input_length)
            for text in swap_seq:
                final_text_list.append(text)
        final_text_list.append(input_sequence)
        return final_text_list

    def test_augment(self, input_sequence, input_length):
        """
        test_augment method is written for augment text for evaluation method
        :param input_sequence: sequence(=sentence) of word index
        :param input_length: input sequence(=sentence) length
        :return:
            output_sequence: list of all augmented sequence(=sentence)
        """
        output_sequence = list()
        # repeat augmentation for 10 times
        for i in range(10):
            augment_text = self.replace_similar_words(input_sequence, input_length)
            for text in augment_text:
                output_sequence.append(text)
        return output_sequence  # contain 20 sentence
