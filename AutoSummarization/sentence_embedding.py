from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import pandas as pd
import numpy as np
import jieba
from gensim.models.word2vec import LineSentence
from functools import reduce
import  networkx
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity





def cut(string):return ' '.join(jieba.cut(string))


def get_news(file_path,stopwords):
    news_content = pd.read_csv(file_path, encoding='gb18030')
    pure_content = pd.DataFrame()
    pure_content['content'] = news_content['content']
    pure_content = pure_content.fillna('')
    pure_content['tokenized_content'] = pure_content['content'].apply(cut)
    reduce(lambda a, b: a + b, pure_content['tokenized_content'].tolist()).split()

    tokeners = [t for l in pure_content['tokenized_content'].tolist() for t in l.split()]

    temp = []
    for word in tokeners:
        if word not in stopwords:
            temp.append(word)

    temp = np.array(temp)

    np.save('./tokeners.npy',temp)
    return  tokeners

def split_sentence(sentences:str,pat = '，。？！,.',minLen = 8):
    #minLen 用于控制句子长度，字符少于minLen的话，就认为句子过于太短，需要和后面句子合并
    splited_sentences = []
    sentences = re.sub(r'[\r\n]', '', sentences)
    sentences = re.split('([，。？！,.])', sentences)
    for sent in sentences:
        if sent not in pat:
            splited_sentences.append(sent)

    #将长度过于短的句子和后面一个句子放在一起
    temp = []
    for i in range(len(splited_sentences)-1):
        a = len(splited_sentences[i])
        if a < minLen:
            splited_sentences[i + 1] = splited_sentences[i] + splited_sentences[i + 1]
    # 这里分两次循环是为了避免找不出来的奇怪的错误
    for sent in splited_sentences:
        if len(sent) >= minLen:
            temp.append(sent)
    splited_sentences  = temp
    return splited_sentences

file_path = './sqlResult_1558435.csv'


def get_stop_words(file: str, encoding='utf-8'):
    ret = [l for l in open(file, encoding=encoding).read()]
    return ret

stopwords = get_stop_words('./stop_words.txt')

def clean_sentences(sents: list, stop_words: list):
    ret = {}
    for i, sent in enumerate(sents):
        words = jieba.lcut(sent)
        wds = []
        for w in words:
            if w not in stop_words:
                wds.append(w)
        if wds:
            ret[i] = wds
    return ret

def sentence_embedding(sentence):
    alpha = 1e-4
    max_fre = max(frequence.values())
    words = cut(sentence).split()
    col = model.wv.vector_size

    sentence_vec = np.zeros(col)

    words = [w for w in words if w in model]
    for w in words:
        if w in model.wv.vocab and w not in stopwords:
            weight = alpha / (alpha + frequence.get(w,max_fre))
            sentence_vec += weight * model.wv[w]
    sentence_vec /= len(words)
    #skip the PCA
    return  sentence_vec

def sentence_similarity_func(sent1,sent2):

    vec1 = sentence_embedding(sent1)
    vec2 = sentence_embedding(sent2)

    ret = 0
    ret = cosine_similarity(vec1.reshape(1,-1), vec2.reshape(1,-1))[0][0]

    return ret

def get_correlations(sents,sentence_embedding,sentence_similarity_func):
    #这里将整个文章拉成一个句子，然后得到整个文章的sentence embedding，再
    #将各个分句子和文章的总的sentence embedding比较，选出和整体比较相关的句子作为提取结果
    text = ' '.join(sents)
    # text_vec = sentence_embedding(text)
    sims = []
    for sent in sents:
        sim = sentence_similarity_func(text,sent)
        sims.append(sim)

    ret = [(sents[ind],sims[ind]) for ind in sorted(range(len(sims)),key = lambda  i:sims[i],reverse=True )]
    #这里返回的就是每个句子与整个文章的相似度的字典，key是每个句子，value是每个句子与文章的相似度
    return ret

def load_obj():
    with open('./frequence.pkl', 'rb') as f:
        return pickle.load(f)

def get_summarization_saimple(text,score_fn,sentence_embedding,sentence_similarity_func,constraint = 100):
    sub_sentence = split_sentence(text)
    ranking_sentence = score_fn(sub_sentence,sentence_embedding,sentence_similarity_func)
    selected_text = set()
    current_text = ''
    for sen,_ in ranking_sentence:
        if len(current_text) < constraint:
            current_text += sen
            selected_text.add(sen)
        else:
            break
    summarized = []
    for sen in sub_sentence:
        if sen in selected_text:
            summarized.append(sen)
    return summarized


text =  '虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，' \
        '但是今年 3 月份官方突然宣布回归中国，预示着很快就有夏普新机在中国登场了。那么，第一' \
        '款夏普手机什么时候登陆中国呢？又会是怎么样的手机呢？\r\n近日，一款型号为 FS8016 的夏' \
        '普神秘新机悄然出现在 GeekBench 的跑分库上。从其中相关信息了解到，这款机子并非旗舰定' \
        '位，所搭载的是高通骁龙 660 处理器，配备有 4GB 的内存。骁龙 660 是高通今年最受瞩目的芯' \
        '片之一，采用 14 纳米工艺，八个 Kryo 260 核心设计，集成 Adreno 512 GPU 和 X12 LTE 调制' \
        '解调器。\r\n当前市面上只有一款机子采用了骁龙 660 处理器，那就是已经上市销售的 OPPO R1' \
        '1。骁龙 660 尽管并非旗舰芯片，但在多核新能上比去年骁龙 820 强，单核改进也很明显，所以' \
        '放在今年仍可以让很多手机变成高端机。不过，由于 OPPO 与高通签署了排他性协议，可以独占两' \
        '三个月时间。\r\n考虑到夏普既然开始测试新机了，说明只要等独占时期一过，夏普就能发布骁龙' \
        ' 660 新品了。按照之前被曝光的渲染图了解，夏普的新机核心竞争优势还是全面屏，因为从 2013 ' \
        '年推出全球首款全面屏手机 EDGEST 302SH 至今，夏普手机推出了多达 28 款的全面屏手机。\r\n在' \
        ' 5 月份的媒体沟通会上，惠普罗忠生表示：“我敢打赌，12 个月之后，在座的各位手机都会换掉。因' \
        '为全面屏时代的到来，我们怀揣的手机都将成为传统手机。'


frequence = load_obj()

model =FastText.load('./news.model')
# print(model['小米'])
# tokeners = np.load('./tokeners.npy')
result = get_summarization_saimple(text,get_correlations,sentence_embedding,sentence_similarity_func,200)
print(result)

















