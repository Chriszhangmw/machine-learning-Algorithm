#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : set_maxLength.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/2/5 19:17
import matplotlib.pyplot as plt
import pandas as pd
import jieba
plt.xlabel('comments content length range')
plt.ylabel('Number of comments')
plt.title('The statistics of news dataset')
plt.xlim(0,1000)
plt.ylim(0,1500)
x = 0
y = {}
content = pd.read_csv('D:\\Eclipse_workplace\\Training\\NLPTraining\\project3\\data\\ai_challenger_sentiment_analysis_trainingset_20180816\\sentiment_analysis_trainingset.csv', encoding='utf-8')
content = content.fillna('')
news_from = content['content']
# print(all_news)
print('一共包含了{}条commets'.format(len(news_from)))
# num = 0
# for title in news_from:
#     if title.strip() == '新华社':
#         num +=1
# print('一共包含了{}条新华社出版的数据'.format(num))
news_content = content['content']
print(len(news_content))
for new in news_content:
    if x%1000 ==0:
        print('have finished {} lines'.format(x))
    x +=1
    line = str(new.strip())
    words = list(jieba.cut(line))
    length = len(words)
    if length in y.keys():
        y[length] +=1
    else:
        y[length] = 1
plt.bar(y.keys(),y.values(),facecolor='green')
plt.show()


# 因此，将每个样本的长度设置为  500（700） 比较合理