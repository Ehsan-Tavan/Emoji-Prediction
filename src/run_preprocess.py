#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=import-error

"""
run_preprocess.py is writen for preprocessing tweets
"""

__author__ = "Ehsan Tavan"
__project__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "10/17/2020"

import time
import random
from itertools import chain
import hazm
import pandas as pd
from emoji_prediction.pre_process.normalizer import Normalizer
from emoji_prediction.tools.log_helper import process_time
from emoji_prediction.config.bilstm_config import RAW_NO_MENTION_DATA_PATH,\
    TRAIN_NORMAL_NO_MENTION_DATA_PATH, TEST_NORMAL_NO_MENTION_DATA_PATH,\
    RAW_DATA_PATH, TRAIN_NORMAL_DATA_PATH, TEST_NORMAL_DATA_PATH


class Cleaning:
    """
    Cleaning class use for normalizing tweets
    """
    def __init__(self, input_path, train_output_path, test_output_path):
        self.normalizer = Normalizer()
        self.input_path = input_path
        self.train_output_path = train_output_path
        self.test_output_path = test_output_path

    def read_tweets(self):
        """
        read_tweets method is writen for read tweets from .csv file
        and tell some information from raw data
        :return:
            raw_tweets: all unique tweets
            raw_labels: the emojis of tweets
        """
        print("Start Reading tweets")
        # load csv file
        data_frame = pd.read_csv(self.input_path)

        # drop duplicates tweets
        data_frame = data_frame.drop_duplicates(subset=["tweets"], keep="first")

        print(f"We have {len(data_frame)} tweets.")
        print(f"We have {len(set(data_frame.labels))} unique emoji type.")

        raw_tweets = data_frame.tweets
        raw_labels = data_frame.labels

        return raw_tweets, raw_labels

    def normalizing(self):
        """
        normalizing method is writen for normalizing tweets
        :return:
            normal_tweets: normal tweets
            emojis: emojis of normal tweets
        """
        # load tweets and emojis
        raw_tweets, raw_emojis = self.read_tweets()
        print("Start normalizing tweets ...")

        start_time = time.time()
        # normalizing tweets
        normal_tweets = [self.normalizer.normalizer_text(tweet) for tweet in raw_tweets]
        end_time = time.time()

        # calculate normalizing time
        elapsed_mins, elapsed_secs = process_time(start_time, end_time)
        print(f"{elapsed_mins} min and {elapsed_secs} sec for normalizing tweets.")
        print("End normalizing tweets")
        return normal_tweets, raw_emojis

    def test_split(self, normal_tweets, normal_emojis):
        """
        test_split method is written for split data into train and test set
        :param normal_tweets: list of all tweets
        :param normal_emojis: list of all emojis
        """
        # shuffle tweets
        tweets_list = list(zip(normal_tweets, normal_emojis))
        random.shuffle(tweets_list)
        random.shuffle(tweets_list)
        normal_tweets, normal_emojis = zip(*tweets_list)

        test_tweet_list = []    # list for test tweets
        test_emoji_list = []    # list for test emojis
        train_tweet_list = []   # list for train tweets
        train_emoji_list = []   # list for train emojis

        # split test tweets
        start_time = time.time()

        for tweet, emoji in zip(normal_tweets, normal_emojis):
            # filter tweets that have no character
            if tweet != "":
                if test_emoji_list.count(emoji) < 2000:
                    test_tweet_list.append(tweet)
                    test_emoji_list.append(emoji)
                else:
                    train_tweet_list.append(tweet)
                    train_emoji_list.append(emoji)

        end_time = time.time()

        # calculate test split time
        elapsed_mins, elapsed_secs = process_time(start_time, end_time)
        print(f"{elapsed_mins} min and {elapsed_secs} sec for test split tweets.")

        # save data
        self.save_normal_tweets(train_tweet_list, train_emoji_list,
                                output_path=self.train_output_path)
        self.save_normal_tweets(test_tweet_list, test_emoji_list,
                                output_path=self.test_output_path)

    @staticmethod
    def save_normal_tweets(normal_tweets, normal_emojis, output_path):
        """
        save_normal_tweets method is writen for save normalized tweets
        :param normal_tweets: all normalized tweets
        :param normal_emojis: all emojis
        :param output_path: output path for save data
        """
        # create dataFrame
        data_frame = pd.DataFrame({"tweets": normal_tweets, "emojis": normal_emojis})

        # save dataFrame
        data_frame.to_csv(output_path, index=False)
        print("Tweets saved.")

    @staticmethod
    def count_chars(list_of_tweets):
        """
        count_chars method is writen for count tweets characters
        :param list_of_tweets: list that contain all tweets
        """
        # get unique characters from tweets
        chars = list(sorted(set(chain(*list_of_tweets))))

        print(f"We have {len(chars)} character")
        print(chars)


class AddPos:
    """
    In this class we add pos to our dataset
    """
    def __init__(self, train_input_path, test_input_path):
        self.pos_tag = hazm.POSTagger(model="../data/Hazm_resources/resources-0.5/postagger.model")
        self.train_input_path = train_input_path
        self.test_input_path = test_input_path

    @staticmethod
    def read_input_file(input_path):
        """
        read_input_file method is written for read input dataFrame
        :param input_path: address of input dataFrame
        :return:
            data_frame: input dataFrame
        """
        # read dataFrame
        data_frame = pd.read_csv(input_path)
        return data_frame

    def make_pos(self, input_text):
        """
        make_pos method is written for tag input sentence
        :param input_text: input_text
        :return:
            output: input text that turn into pos tag
        """
        # pos tagging with hazm library
        pos_sen = self.pos_tag.tag(hazm.word_tokenize(input_text))
        output = ""
        for pos in pos_sen:
            output = output + " " + pos[1]
        return output.strip()

    def make_dataset(self, input_path):
        """
        make_dataset method is written for add pos column to input dataFrame
        :param input_path: address of input dataFrame
        """
        # read dataFrame
        data_frame = self.read_input_file(input_path)
        pos_list = []
        for tweet in data_frame.tweets:
            pos_list.append(self.make_pos(tweet))

        # adding pos column to dataFrame
        data_frame["pos"] = pos_list

        # save dataFrame with pos
        data_frame.to_csv(input_path, index=False)

    def __run__(self):
        """
        __run__ method is written for running AddPos class
        """
        # adding pos to train data
        self.make_dataset(self.train_input_path)
        # adding pos to test data
        self.make_dataset(self.test_input_path)


if __name__ == "__main__":
    POSCLASS = AddPos(train_input_path=TRAIN_NORMAL_DATA_PATH,
                      test_input_path=TEST_NORMAL_DATA_PATH)
    POSCLASS.__run__()

    # CLEANING_CLASS = Cleaning(input_path=RAW_DATA_PATH,
    #                           train_output_path=TRAIN_NORMAL_DATA_PATH,
    #                           test_output_path=TEST_NORMAL_DATA_PATH)
    # TWEETS, EMOJIS = CLEANING_CLASS.normalizing()
    # CLEANING_CLASS.test_split(TWEETS, EMOJIS)
