#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : data_process.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/2/5 19:32
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import tensorflow as tf
from keras import optimizers,regularizers
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D,BatchNormalization,Conv2D,MaxPooling2D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import numpy as np
import jieba
import  re
from string import digits
import matplotlib.pyplot as plt
import  pandas  as pd
from project3.util import under_sample



def jieba_cut(string):
    words = list(jieba.cut(string))
    ss = ' '.join(words)
    return ss

content = pd.read_csv(
        'D:\\Eclipse_workplace\\Training\\NLPTraining\\project3\\data\\ai_challenger_sentiment_analysis_trainingset_20180816\\sentiment_analysis_trainingset.csv',
        encoding='utf-8')
content = content.fillna('')
data_for_keras = np.array(content['content'])

def process_train(tag):
    print('start to process trainning dataSet ')
    x_train = []
    y_train = []
    content = pd.read_csv(
        'D:\\Eclipse_workplace\\Training\\NLPTraining\\project3\\data\\ai_challenger_sentiment_analysis_trainingset_20180816\\sentiment_analysis_trainingset.csv',
        encoding='utf-8')
    print('start to process undersample the trainning dataSet ')
    content = under_sample(content,tag)
    print('finish to process undersample the trainning dataSet ')
    content = content.fillna('')
    columns = content.columns.values.tolist()
    label_index = columns.index(tag)
    for index in content.index:
        line = content.loc[index].values
        comments = line[1]
        comments = comments.replace(' ','')
        comments = comments.replace('\n', '')
        wordlist_string = jieba_cut(str(comments))
        x_train.append(wordlist_string)
        y_train.append(line[label_index])
    print('start to use keras tool to transfor trainning dataSet ')
    tokenizer = Tokenizer(filters='!"#$%&()*+,。，；-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", num_words=15000)
    tokenizer.fit_on_texts(data_for_keras)
    # vocab = tokenizer.word_index

    X_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(X_train_word_ids, maxlen=700, padding='post')
    print('finish to use keras tool to transfor trainning dataSet ')

    #将y转为one hot的形式
    temp = np.zeros((len(X_train_word_ids),4))
    for index,value in enumerate(y_train):
        if value == 1:
            temp[index][0] = 1
        elif value == 0:
            temp[index][1] = 1
        elif value == -1:
            temp[index][2] = 1
        elif value == -2:
            temp[index][3] = 1
    y_train = temp

    return np.array(x_train),np.array(y_train)

def process_test(tag):
    x_train = []
    y_train = []

    content = pd.read_csv(
        'D:\\Eclipse_workplace\\Training\\NLPTraining\\project3\\data\\ai_challenger_sentiment_analysis_validationset_20180816\\sentiment_analysis_validationset.csv',
        encoding='utf-8')
    content = content.fillna('')
    columns = content.columns.values.tolist()
    label_index = columns.index(tag)
    for index in content.index:
        line = content.loc[index].values
        comments = line[1]
        comments = comments.replace(' ','')
        comments = comments.replace('\n', '')
        wordlist_string = jieba_cut(str(comments))
        x_train.append(wordlist_string)
        y_train.append(line[label_index])

    tokenizer = Tokenizer(filters='!"#$%&()*+,。，；-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", num_words=15000)
    tokenizer.fit_on_texts(data_for_keras)
    # vocab = tokenizer.word_index

    X_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(X_train_word_ids, maxlen=700, padding='post')

    temp = np.zeros((len(X_train_word_ids)))
    for index, value in enumerate(y_train):
        if value == 1:
            temp[index] = 0
        elif value == 0:
            temp[index] = 1
        elif value == -1:
            temp[index] = 2
        elif value == -2:
            temp[index] = 3
    y_train = temp

    return np.array(x_train),np.array(y_train)




































