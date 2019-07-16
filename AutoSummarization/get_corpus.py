#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : get_corpus.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/3/18 15:43


import pandas as pd
import os
import re
import numpy as np
import jieba

# news_data = pd.read_csv('./sqlResult_1558435.csv', encoding='gb18030')
#
# news = news_data['content']
# with open('./corpus.txt','w',encoding='utf-8') as f:
#     for new in news:
#         new = str(new).strip()
#         f.write(new)
#     f.close()

def cut(string):return ' '.join(jieba.cut(string))

file_path = './sqlResult_1558435.csv'
news_content = pd.read_csv(file_path, encoding='gb18030')
pure_content = pd.DataFrame()
pure_content['content'] = news_content['content']
pure_content = pure_content.fillna('')
pure_content['tokenized_content'] = pure_content['content'].apply(cut)
with open('corpus_cutted.txt','w',encoding='utf-8') as f:
    f.write(' '.join(pure_content['tokenized_content'].tolist()))

















