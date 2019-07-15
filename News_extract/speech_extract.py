# 步骤：
# 提取“说”的近义词（Word2Vec + 图搜索）；
# 使用LTP进行词性标注、依存关系分析，获取中心词。判断中心词是否在“说”的近义词列表里。是，则下一步；
# 寻找中心词对应的命名实体；
# 提取可能是人物言论的分句，以句号、问号或感叹号作为分句标志；
# 使用词向量加权和PCA降维的方法计算句子向量；
# 计算第二个及之后的句子向量与第一个句子向量的余弦值，判断句子之间的相似程度，获得人物言论的结束分句。

import linecache
import re
import time
from collections import Counter
from collections import defaultdict

import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from News_Extraction.wangkun.ltp_api import MyLtp  # 导入ltp_api.py中的接口


# 加载训练好的维基W2V模型
def get_model_from_file(filename):
    model = Word2Vec.load(filename)
    return model


# 根据图搜索获得词的相关词及相关性
def get_related_words(word, model, max_size):
    start = [word]
    seen = defaultdict(int)
    while len(seen) < max_size:
        cur = start.pop(0)
        seen[cur] += 1
        for w, r in model.wv.most_similar(cur, topn=20):
            seen[w] += 1
            start.append(w)
    return seen


# 将新闻语料预处理写入文件
def write_news_corpus_to_file(news, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(len(news)):
            try:
                co = news['content'][i].replace('\r\n', '\n')
                co = co.replace('\\n', '\n')  # characters '\n' in text
                for line in co.split('\n'):
                    l = line.strip()
                    if l:
                        li = jieba.lcut(l)
                        f.write(' '.join(li))
                        f.write('\n')
            except:
                # print(news['content'][i]) # 'nan'
                pass


# 在训练好的模型上增加更多训练预料训练
def add_more_train_corpus(model, filename):
    sen_list = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            sen_list.append(line.split())
    model.train(sentences=sen_list, total_examples=len(sen_list), epochs=1)


# 从新闻语料中获得词频率，后面词向量加权得到句子向量用到
def word_freq(corpus_file):
    word_list = []
    with open(corpus_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            word_list += line.split()
    cc = Counter(word_list)
    num_all = sum(cc.values())
    def get_word_freq(word):
        return cc[word] / num_all
    return get_word_freq


# 移除分句中的特殊符号
def rm_spec(word_list):
    ret = []
    for w in word_list:
        wr = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——\-()?【】《》“”！，。？、~@#￥%……&*（）]+', '', w)
        if wr:
            ret.append(wr)
    return ret


# 获得句子向量矩阵
def get_sentences_vec(model, sent_list, get_wd_freq):
    # 词向量加权部分
    a = 0.001
    row = model.wv.vector_size
    col = len(sent_list)
    sent_mat = np.mat(np.zeros((row, col)))
    for i, sent in enumerate(sent_list):
        # new_sent = rm_spec(sent)
        new_sent = sent
        if not new_sent: continue
        sent_vec = np.zeros(row)
        for word in new_sent:
            pw = get_wd_freq(word)
            w = a / (a + pw)
            try:
                vec = np.array(model.wv[word])
                sent_vec += w * vec
            except:
                pass
        sent_mat[:, i] += np.mat(sent_vec).T
        sent_mat[:, i] /= len(new_sent)

    # 减去PCA中的第一主成分
    u, s, vh = np.linalg.svd(sent_mat)
    sent_mat = sent_mat - u * u.T * sent_mat
    return sent_mat


# 计算余弦相似度
def get_cos_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0
    return np.sum(np.array(v1) * np.array(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 返回句子向量矩阵中各列向量与第一列向量的相似度
def get_sent_vec_sims(sent_mat):
    first = np.array(sent_mat[:, 0])[:, 0]
    col = sent_mat.shape[1]
    sims = [1.0]
    for i in range(1, col):
        vec = np.array(sent_mat[:, i])[:, 0]
        sims.append(get_cos_similarity(first, vec))
    return sims


# 获得最终说的话
def get_final_sents(sents, sims):
    senlist = []
    threshold = 0.5  # 相似度超过0.5即认为两句话相关
    for i in range(len(sims)):
        if sims[i] > threshold:
            senlist.append(sents[i])
    ret = [''.join(s) for s in senlist]
    return ret


if __name__ == '__main__':
    w2v_model = get_model_from_file('./wiki_w2v.model')
    w2v_model.wv.most_similar('说道', topn=100)

    # news_df = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
    # write_news_corpus_to_file(news_df, 'news_corpus.txt')

    start_time = time.time()
    add_more_train_corpus(w2v_model, './news_corpus.txt')
    print('elapsed time: {}'.format(time.time() - start_time))
    w2v_model.wv.most_similar('说道', topn=100)

    related_words = get_related_words('说道', w2v_model, 500)
    related = sorted(related_words.items(), key=lambda x: x[1], reverse=True)
    print('related\n', related)

    similar_words = filter(lambda x: x[1] > 2, related)
    talk_sims = []
    for w, c in similar_words:
        talk_sims.append(w)
        print('{}\t{}'.format(w, c))

    get_word_frequency = word_freq('./news_corpus.txt')

    all_lines = linecache.getlines('./news_corpus.txt')

    my_ltp = MyLtp()

    with open('./speech.csv', 'a', encoding='gb18030') as fout:
        for ll in range(0, len(all_lines)):
            if ll % 1000 == 0:
                print('processed: ', ll)

            line = all_lines[ll]
            lc = []
            if len(line) > 1000: # 处理句子过长的情况
                lc = line.split('。')
            else:
                lc = [line]

            for li in lc:
                words = li.split()
                if len(words) == 0: continue

                ne, talk, sents = my_ltp.get_character_speech(words, talk_sims)

                if len(sents) != 0:
                    # print('ne:', ne)
                    # print('talk:', talk)
                    # print('sents:', sents)
                    sims = [1.0]
                    if len(sents) > 1:
                        sent_mat = get_sentences_vec(w2v_model, sents, get_word_frequency)
                        sims = get_sent_vec_sims(sent_mat)
                        # print(sims)

                    final_sents = get_final_sents(sents, sims)
                    content = ''.join(final_sents)
                    if len(content) > 0:
                        fout.write(ne)
                        fout.write('----->')
                        fout.write(talk)
                        fout.write('----->')
                        fout.write(content)
                        fout.write('\n')
    my_ltp.clean()
