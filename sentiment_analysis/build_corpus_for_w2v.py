#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : build_corpus_for_w2v.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/3/18 11:13
import numpy as np
import pandas as pd
train_data = pd.read_csv('./trainset.csv')
sentence_list = train_data['content']
with open('./corpus.txt','w',encoding='utf-8') as f:
    for line in sentence_list:
        f.write(line.strip() + '\n')
    f.close()