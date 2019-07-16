#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Free MMan
# @Site    : https://github.com
# @File    : text_abstract.py
# @Software: PyCharm Professional Edition
# @Time    : 2019/3/18 14:10
import pandas as pd
import os
import re
import numpy as np
from collections import Counter
from gensim.models import FastText
import jieba
import networkx as nx
import math
from gensim.models.word2vec import LineSentence
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial

news_data = pd.read_csv('./sqlResult_1558435.csv', encoding='gb18030')
news = news_data['content'][47]

# print(news)

#这个方法根据标点符号，将文章分为各个句子，同事得到标点符号和句子的list的index
def split_sentence(sent):
    sent = re.sub(r'[\r\n]', '', sent)
    ls = re.split('([，。？！,.])', sent)
    pat = '，。？！,.'
    sent_ids = []
    symbol_ids = []
    for i, s in enumerate(ls):
        if s in pat:
            symbol_ids.append(i)
        else:
            sent_ids.append(i)
    return ls, sent_ids, symbol_ids

def get_stop_words(file: str, encoding='utf-8'):
    ret = [l for l in open(file, encoding=encoding).read()]
    return ret

def clean_sentence(sents:list,stop_words:list):
    ret = {}
    for i ,sent in enumerate(sents):
        words = jieba.lcut(sent)
        wds = []
        for w in words:
            if w not in stop_words:
                wds.append(w)
        if wds:
            ret[i] = wds
    return ret


def sentence_similarity_1(sent1,sent2):
    ret = 0
    same_cnt = 0
    for w in sent1:
        if w in sent2:
            same_cnt +=1
    ret = same_cnt / (math.log2(len(sent1)) + math.log2(len(sent2)) + 1)
    return ret


# ls, sent_ids, symbol_ids = split_sentence('上周周末，中超第13轮。已经全部！战罢，其中，广州恒大？战胜贵州恒。丰智诚，豪取联赛九连，胜的同时继续领跑积分榜')

def sentence_similarity_func(model,prob_func):
    a = 0.001
    col = model.wv.vector_size

    def core(sent1,sent2):
        vec1 = np.zeros(col)
        vec2 = np.zeros(col)
        for w in sent1:
            pw = a / (a + prob_func(w))
            vec1 += pw * model.wv[w]
        for w in sent2:
            pw = a / (a + prob_func(w))
            vec2 += pw * model.wv[w]
        ret = 0
        set = cosine_similarity(vec1.reshape(1,-1),vec2.reshape(1,-1))[0][1]
        return ret
    return  core

def sentence_embedding(model,prob_func,stop_words):
    a = 0.01
    col = model.wv.vector_size

    def core(sent):
        vec = np.zeros(col)
        words = jieba.lcut(sent)
        for w in words:
            if w in model.wv.vocab and w not in stop_words:
                pw = a / (a + prob_func(w))
                vec += pw * model.wv[w]
        return vec
    return core


def get_prob(counter):
    total_cnt = sum(counter.values())
    def core(word:str):
        return counter[word]/total_cnt
    return  core

#给定一个字典，计算字典各个句子之间的相似性
def gen_sim_mat(sent_dict:dict,sim_func):
    size = len(sent_dict)
    ret = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            if i!=j :
                ret[i][j] = sim_func(sent_dict[i],sent_dict[j])
    return ret



def get_final_sentence(sent_list,indices,word_cnt=200):
    cur_cnt = 0
    lst = []
    for i in indices:
        sent = sent_list[i]
        size = len(sent)
        cur_cnt += size
        if cur_cnt < word_cnt:
            lst.append(i)
    ret = '。'.join([sent_list[i] for i in lst])
    return  ret
stop_words = get_stop_words('./stop_words.txt')


def get_correlations(sents,sent_vec_func):
    text = ' '.join(sents)
    text_vec = sent_vec_func(text)
    sims = []
    for sent in sents:
        vec = sent_vec_func(sent)
        sim = cosine_similarity(vec.reshape(1,-1),text_vec.reshape(1,-1))[0][1]
        sims.append(sim)
    ret = [(sents[ind],sims[ind]) for ind in sorted(range(len(sims)),key = lambda i:sims[i] ,reverse=True)]
    return ret

from gensim.models import Word2Vec
# Load trained w2v model
def load_w2v_model(filename):
    model = Word2Vec.load(filename)
    return model


tokeners = []
for line in open('./corpus.txt', 'r', encoding='utf-8'):
    tokeners += line.split()


model = load_w2v_model('./w2v.model')
# print(len(tokeners))
# print(model.wv['小米'])

word_counter = Counter(tokeners)

prob_func = get_prob(word_counter)

get_sentence_vec = sentence_embedding(model, prob_func, stop_words)
# ls, sent_ids, symbol_ids = split_sentence(news)

# print(get_sentence_vec('小米'))

def get_summarization_simple(text, score_fn, constraint=200):
    sents, sent_ids, symb_ids = split_sentence(text)
    sub_sentence = [sents[i] for i in sent_ids]
    ranking_sentence = score_fn(sub_sentence)
    selected_text = set()
    current_text = ''

    for sen, _ in ranking_sentence:
        if len(current_text) < constraint:
            current_text += sen
            selected_text.add(sen)
        else:
            break

    summarized = []
    for sen in sub_sentence:  # print the selected sentence by sequent
        if sen in selected_text:
            ind = sents.index(sen)
            summarized.append(sen)
            if (ind + 1) in symb_ids:
                summarized.append(sents[ind + 1])
    return ''.join(summarized)

score_func_embed = partial(get_correlations, sent_vec_func=get_sentence_vec)

# result = get_summarization_simple(news, score_func_embed, 200)
# print(result)


def get_connect_graph_by_text_rank(tokenized_text,window=3):
    keywords_graph = nx.Graph()
    tokeners = tokenized_text
    print(tokeners)
    for ii,t in enumerate(tokeners):
        word_tuples = [(tokeners[connect],t) for connect in range(ii-window,ii+window+1)
            if connect >= 0 and connect < len(tokeners)]
        keywords_graph.add_edges_from(word_tuples)
    return keywords_graph

def sentence_ranking_by_text_ranking(split_sentence):
    sentence_graph = get_connect_graph_by_text_rank(split_sentence)
    ranking_sentence = nx.pagerank(sentence_graph)
    ranking_sentence = sorted(ranking_sentence.items(),key = lambda x: x[1], reverse=True)
    print(ranking_sentence)
    return  ranking_sentence


def get_summarization_simple_with_text_rank(text,constraint = 200):
    return get_summarization_simple(text,sentence_ranking_by_text_ranking, constraint)


result = get_summarization_simple_with_text_rank(news,constraint=200)
print(result)









