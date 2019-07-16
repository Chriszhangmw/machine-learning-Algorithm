#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : textCNN.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/2/5 20:08

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
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import numpy as np

import matplotlib.pyplot as plt
import  pandas  as pd


'''
'location_traffic_convenience'
'location_distance_from_business_district' 'location_easy_to_find'
'service_wait_time' 'service_waiters_attitude'
'service_parking_convenience' 'service_serving_speed' 'price_level'
'price_cost_effective' 'price_discount' 'environment_decoration'
'environment_noise' 'environment_space' 'environment_cleaness'
'dish_portion' 'dish_taste' 'dish_look' 'dish_recommendation'
'others_overall_experience' 'others_willing_to_consume_again'
'''


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=None)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="blue" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    class_names = ['pos','neu', 'neg','notme']
    plot_confusion_matrix(confusion_matrix
                          , classes=class_names
                          , title='Confusion matrix')



def text_cnn(maxlen=700, max_features=15000, embed_size=32):
    conment_seq = Input(shape=[maxlen], name='x_seq')
    emb_comment = Embedding(max_features, embed_size)(conment_seq)
    convs = []
    filter_sizes = [2,3, 4,5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.3)(merge)
    output = Dense(32, activation='relu')(out)
    # activity_regularizer = regularizers.l1(0.01),kernel_regularizer = regularizers.l2(0.01)

    output = Dense(units=4, activation='softmax')(output)
    model = Model([conment_seq], output)
    a = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss="categorical_crossentropy", optimizer=a, metrics=['accuracy'])
    return model
from project3.data_process import process_train
from project3.data_process import process_test

x_train ,y_train = process_train('location_distance_from_business_district')
x_test,y_test = process_test('location_distance_from_business_district')
model_ = text_cnn(700,15000,32)
model_.summary()

history = model_.fit(x_train, y_train,
          batch_size=32,
          epochs=100,
          validation_split=0.1)
y = model_.predict(x=x_test)
y_pre = []
for pre in y:
    pre = np.array(pre)
    y_pre.append(np.argmax(pre))
print('*'*20)
print(y_test)
print(len(y_test))
print(y_pre)
print(len(y_pre))

plot_matrix(y_test,y_pre)

# score = model_.evaluate(x=x_test, y=y_test, verbose=1)
# print("test loss:", score[0])
# print("test acc:", score[1])
#
# # display the change of parameter during the traning process
# print(history.history.keys())
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title("model accuracy")
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper left")
# plt.show()
#
# # summary the loss value and acc value
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title("model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["train", "test"], loc="upper left")
# plt.show()






