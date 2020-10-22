#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
log_helper.py is a file to write methods which use for better log
"""

import logging

__author__ = "Ehsan Tavan"
__organization__ = "Persian Emoji Prediction"
__credits__ = ["Ehsan Tavan"]
__license__ = "Public Domain"
__version__ = "1.1.1"
__maintainer__ = "Ehsan Tavan"
__email__ = "tavan.ehsan@gmail.com"
__status__ = "Production"
__date__ = "10/09/2020"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def count_parameters(input_model):
    """
    count_parameters method is written for calculate number of model's parameter
    :param input_model: model
    :return:
        num_parameters: number of model parameters
    """
    num_parameters = sum(p.numel() for p in input_model.parameters() if p.requires_grad)
    return num_parameters


def process_time(s_time, e_time):
    """
    process_time method is written for calculate time
    :param s_time: start time
    :param e_time: end time
    :return:
        elapsed_mins: Minutes of process
        elapsed_secs: Seconds of process
    """
    elapsed_time = e_time - s_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def model_result_log(train_log_dict, valid_log_dict, test_log_dict):
    """
    model_result_log method is writen for show model result on each epoch
    :param train_log_dict: dictionary of train data result
    :param valid_log_dict: dictionary of validation data result
    :param test_log_dict: dictionary of test data result
    """
    logging.info(f"\tTrain. Loss: {train_log_dict['train_loss']:.4f} | "
                 f"Train. Acc: {train_log_dict['train_acc'] * 100:.2f}%")
    logging.info(f"\t Val. Loss: {valid_log_dict['valid_loss']:.4f} |  "
                 f"Val. Acc: {valid_log_dict['valid_acc'] * 100:.2f}%")
    logging.info(f"\t Test. Loss: {test_log_dict['test_loss']:.4f} |  "
                 f"Test. Acc: {test_log_dict['test_acc'] * 100:.2f}%")

    logging.info(f"\t Train. Precision: {train_log_dict['train_precision']}")
    logging.info(f"\t Train. Recall: {train_log_dict['train_recall']}")
    logging.info(f"\t Train. F1_Score: {train_log_dict['train_fscore']}")
    logging.info(f"\t Train. Total F1 score: {train_log_dict['train_total_fscore']}")

    logging.info(f"\t Val. Precision: {valid_log_dict['valid_precision']}")
    logging.info(f"\t Val. Recall: {valid_log_dict['valid_recall']}")
    logging.info(f"\t Val. F1_Score: {valid_log_dict['valid_fscore']}")
    logging.info(f"\t Val. Total F1 score: {valid_log_dict['valid_total_fscore']}")

    logging.info(f"\t Test. Precision: {test_log_dict['test_precision']}")
    logging.info(f"\t Test. Recall: {test_log_dict['test_recall']}")
    logging.info(f"\t Test. F1_Score: {test_log_dict['test_fscore']}")
    logging.info(f"\t Test. Total F1 score: {test_log_dict['test_total_fscore']}")

    logging.info("_____________________________________________________________________\n")


def model_result_save(log_file, train_log_dict, valid_log_dict, test_log_dict):
    """
    model_result_save method is writen for save model result on each epoch
    :param train_log_dict: dictionary of train data result
    :param valid_log_dict: dictionary of validation data result
    :param test_log_dict: dictionary of test data result
    :param human_test_log_dict: dictionary of human test data result
    """
    # log_file.write(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n")
    log_file.write(f"\tTrain Loss: {train_log_dict['train_loss']:.4f} | "
                   f"Train Acc: {train_log_dict['train_acc'] * 100:.2f}%\n")
    log_file.write(f"\t Val. Loss: {valid_log_dict['valid_loss']:.4f} |  "
                   f"Val. Acc: {valid_log_dict['valid_acc'] * 100:.2f}%\n")
    log_file.write(f"\t Test. Loss: {test_log_dict['test_loss']:.4f} |  "
                   f"Test. Acc: {test_log_dict['test_acc'] * 100:.2f}%\n")

    log_file.write(f"\t Train. Precision: {train_log_dict['train_precision']}\n")
    log_file.write(f"\t Train. Recall: {train_log_dict['train_recall']}\n")
    log_file.write(f"\t Train. F1_Score: {train_log_dict['train_fscore']}\n")
    log_file.write(f"\t Train. Total F1 score: {train_log_dict['train_total_fscore']}\n")

    log_file.write(f"\t Val. Precision: {valid_log_dict['valid_precision']}\n")
    log_file.write(f"\t Val. Recall: {valid_log_dict['valid_recall']}\n")
    log_file.write(f"\t Val. F1_Score: {valid_log_dict['valid_fscore']}\n")
    log_file.write(f"\t Val. Total F1 score: {valid_log_dict['valid_total_fscore']}\n")

    log_file.write(f"\t Test. Precision: {test_log_dict['test_precision']}\n")
    log_file.write(f"\t Test. Recall: {test_log_dict['test_recall']}\n")
    log_file.write(f"\t Test. F1_Score: {test_log_dict['test_fscore']}\n")
    log_file.write(f"\t Test. Total F1 score: {test_log_dict['test_total_fscore']}\n")

    log_file.write("____________________________________________________________\n")
    log_file.flush()
