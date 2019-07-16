from gensim.models import FastText
import pandas as pd
import numpy as np
import jieba
from gensim.models.word2vec import LineSentence
from functools import reduce
import  networkx
import re

# model = FastText(LineSentence('./corpus_cutted.txt'), window=5, size=35, iter=10, min_count=1)


def cut(string):return ' '.join(jieba.cut(string))


def get_news(file_path):
    news_content = pd.read_csv(file_path, encoding='gb18030')
    pure_content = pd.DataFrame()
    pure_content['content'] = news_content['content']
    pure_content = pure_content.fillna('')
    pure_content['tokenized_content'] = pure_content['content'].apply(cut)

    # tokeners = [t for l in pure_content['tokenized_content'].tolist() for t in l.split()]

    return  pure_content

def split_sentence(sentences:str,pat = '，。？！,.',minLen = 8):
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
# pure_content = get_news(file_path)
# text = pure_content.iloc[6]['content']
# print(text)

text =  '在城中村租房作为卖淫场所，组织闲散人员望风并接送卖淫女，恐吓强迫一名12岁幼女卖淫30余次！今日' \
        '（28日），西安未央区法院依法对被告人温某、辛某组织卖淫及被告人郭某协助组织卖淫案进行不公开' \
        '审理，当庭宣判。2017年9月，温某、辛某在西安市未央区东杨善村租用民房作为组织卖淫活动场所，由' \
        '李某和王某（二人因犯强迫卖淫罪已被判处刑罚）等人物色卖淫女，温某、辛某组织张某等三人（均因协' \
        '助组织卖淫罪被判处刑罚）等社会闲散人员望风及接送卖淫人员，温某、辛某从中获利并给以上人员发' \
        '放报酬，30岁的女子郭某协助二人对卖淫女及嫖资进行管理，形成以温某、辛某为首的恶势力团伙。同年9月' \
        '25日下午，李某、王某前往彬县，通过袁某（案发时14周岁）将其朋友小玲（化名，案发时12周岁）约至彬' \
        '县某宾馆并阻止其离开。9月27日，李某、王某和袁某将小玲带至温某、辛某经营的卖淫场所。李某、王某通' \
        '过劝说和恐吓等方式，强迫小玲在此卖淫30余次。期间，温某、辛某安排张某等三人为卖淫窝点望风、负责' \
        '接送看管小玲及其他多名卖淫女。2017年10月10日，小玲的家属以需要“服务”为由，通过交友软件联系到张' \
        '某，并约到丰庆公园附近一家酒店见面。张某在送小玲前往酒店时被小玲的亲友控制并报警。2018年6月11' \
        '日，辛某、郭某被公安机关抓获。8月2日，温某被抓获归案。　　据了解，该卖淫场所取名为“红房子”，组' \
        '织的卖淫女至少有5人，包括小玲等至少两名未成年人。未央法院经审理认为，被告人温某、辛某为牟取非法' \
        '利益，采用招募、雇佣等手段，管理他人卖淫，且卖淫人员在3人以上，已构成组织卖淫罪。被告人郭某负责' \
        '协助管理卖淫女和收取嫖资，已构成协助组织卖淫罪。温某、辛某组织未成年人卖淫，依法应从重处罚。温某' \
        '曾因故意犯罪被判处有期徒刑，又犯本罪，系累犯，应从重处罚。被告人郭某能如实供述主要犯罪事实，并当庭' \
        '表示自愿认罪，可依法从轻处罚。28日，未央法院一审以组织卖淫罪，判处被告人温某有期徒刑8年6个月，并处' \
        '罚金5万元；判处辛某有期徒刑7年，并处罚金5万元；以协助组织卖淫罪，判处郭某有期徒刑2年6月，并处罚' \
        '金1万元。宣判时，未央区委常委、区委政法委书记张瑞斌，未央法院院长李社武，未央检察院检察长胡晓静，未' \
        '央区关工委以及区人大代表、政协委员到庭旁听。'

def get_connect_graph(tokenized_text: str, window=3):
    sentence_grapg = networkx.Graph()
    for ii, t in enumerate(tokenized_text):
        word_tuples = [(tokenized_text[connect], t)
                       for connect in range(ii-window, ii+window+1)
                       if connect >= 0 and connect < len(tokenized_text)]
        # print(word_tuples)
        sentence_grapg.add_edges_from(word_tuples)

    return sentence_grapg






def sentence_ranking_by_text_ranking(split_sentence):
    #拿到所有句子组成的graph，这里认为前后window个窗口之间的句子，是可以互相连接为一个点的
    sentence_grapg = get_connect_graph(split_sentence)
    #根据得到的句子的graph，自动算出各个句子的page value值，最为后续句子重要性的判断
    ranking_sentence = networkx.pagerank(sentence_grapg)

    ranking_sentence = sorted(ranking_sentence.items(),key=lambda x: x[1], reverse=True)

    return  ranking_sentence




def get_summarization_saimple(text,score_fn,constraint = 200):
    sub_sentence = split_sentence(text)
    ranking_sentence = score_fn(sub_sentence)
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

result = get_summarization_saimple(text,sentence_ranking_by_text_ranking,200)
print(result)


# print(tokeners[:10])






















