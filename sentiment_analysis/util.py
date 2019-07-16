#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : static_class_num.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/2/5 16:19

import  pandas as pd
import numpy as np


def under_sample(data,label):
    number_pos = len(data[data[label] == 1])
    number_neu = len(data[data[label] == 0])
    number_neg = len(data[data[label] == -1])
    number_notMention = len(data[data[label] == -2])

    min_number = min(number_pos, number_neu, number_neg, number_notMention)

    pos_indexs = data[data[label] == 1].index
    neu_indexs = data[data[label] == 0].index
    neg_indexs = data[data[label] == -1].index
    notMention_indexs = data[data[label] == -2].index

    random_pos_indices = np.array(np.random.choice(pos_indexs, min_number, replace=False))
    random_neu_indices = np.array(np.random.choice(neu_indexs, min_number, replace=False))
    random_neg_indices = np.array(np.random.choice(neg_indexs, min_number, replace=False))
    random_notMention_indices = np.array(np.random.choice(notMention_indexs, min_number, replace=False))

    final_indexs = np.concatenate([random_pos_indices, random_neu_indices, random_neg_indices, random_notMention_indices])

    under_sample_data = data.iloc[final_indexs, :]

    return under_sample_data

# train_data_path = './ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
# validation_data_path = './ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
# test_data_path = './ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'
#
# data = pd.read_csv(train_data_path)
# # data = under_sample(train_data_path)
# labels = data.columns.values[2:]
# # print(len(labels))
# for label in labels:
#     undersample_data = under_sample(data,label)
#     print('statistics label is ',label)
#     number_pos = len(undersample_data[undersample_data[label] == 1])
#     number_neu = len(undersample_data[undersample_data[label] == 0])
#     number_neg = len(undersample_data[undersample_data[label] == -1])
#     number_notMention = len(undersample_data[undersample_data[label] == -2])
#     print('in the tranning dataSet the {} has {} postive samples,{} Neutral samples,{} negative samples,{} not mentioned samples'.format(label,number_pos,number_neu,number_neg,number_notMention) + '\n')

# 从这里可以统计出，在20个小类别中4个指标的均衡性，可以看到大部分的小类别中，not mention比较多，甚至是其他类别（例如积极）的十多倍，因此不能直接做训练
#第一个想到的是做undersample



