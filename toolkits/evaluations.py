#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
# @Time     : 2018/11/25 10:15
# @Author   : Yiwen Liao
# @File     : evaluations.py 
# @Software : PyCharm
# @Contact  : yiwen.liao93@gmail.com


import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


def cal_openset_baccu(ground_truth=None, prediction=None, label_ref=None):
    """Calculate balanced accuracy for open set recognition.

    :param ground_truth: True labels.
    :param prediction: Predicted labels.
    :param label_ref: A list of class labels in ascending order.
    :return: Balanced accuracy.
    """

    # Abnormal samples have the label of zero which are considered negative
    matrix = confusion_matrix(ground_truth, prediction, labels=label_ref)

    # Number of correctly predicted abnormal samples
    tn = matrix[0, 0]

    # Number of correctly predicted normal samples
    tp = np.trace(matrix) - tn

    num_pos = np.count_nonzero(ground_truth)
    num_neg = len(ground_truth) - num_pos

    tnr = tn/num_neg
    tpr = tp/num_pos
    baccu = 0.5 * (tnr + tpr)

    return baccu


def cal_closed_set_accu(ground_truth=None, prediction=None):
    """Calculate conventional closed set accuracy.

    :param ground_truth: True labels.
    :param prediction: Predicted labels.
    :return: Closed set accuracy.
    """

    closed_set_accuracy = accuracy_score(y_true=ground_truth, y_pred=prediction)
    print('Closed-set accuracy is %.4f' % closed_set_accuracy)

    return closed_set_accuracy


def cal_modified_auc(ground_truth=None, prediction=None):
    """Calculate modified AUC according to Neal et al.

    :param ground_truth: True labels.
    :param prediction: Predicted logits values.
    :return: Modified AUC.
    """

    pred_abnormal = prediction[:, 0]
    pred_normal = np.max(prediction[:, 1:], axis=-1)
    pred_score = pred_abnormal - pred_normal

    auc = roc_auc_score((ground_truth == 0)*1, pred_score)
    print('AUC is %.4f' % auc)

    return auc

