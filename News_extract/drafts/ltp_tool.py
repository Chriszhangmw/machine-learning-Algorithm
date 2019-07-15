from pyltp import Segmentor
from pyltp import Postagger
import jieba
import pyltp
from pyltp import NamedEntityRecognizer
import os
import re


LTP_DATA_DIR = 'E:\\PythonProjects\\Training\\NLPTraining\\project01\\model\\ltp-models'

class LTP():
    def __init__(self):
        self.cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        self.pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        self.ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        self.par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')# 依存句法分析模型路径，模型名称为`parser.model`
        self.key_words = ['表示','指出','认为','坦言','看来','透露','介绍','明说','说','强调','所说','提到','说道','称','声称','建议','呼吁',
              '提及','地说','直言','普遍认为','批评','重申','提出','明确指出','觉得','宣称','猜测','特别强调','写道','引用','相信',
              '解释','谈到','深知','称赞','感慨','主张','还称','中称','指责','披露','明确提出','描述','提醒','深有体会','爆料',
              '裁定','宣布']

    # 移除分句中的特殊符号
    def rm_spec(self,word_list):
        ret = []
        for w in word_list:
            wr = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——\-()?【】《》“”！，。？、~@#￥%……&*（）]+', '', w)
            if wr:
                ret.append(wr)
        return ret

    def cut(self,string):
        segmentor = Segmentor()
        segmentor.load(self.cws_model_path)
        words = segmentor.segment(string)
        temp = []
        for word in words:
            temp.append(word)
        # words = self.rm_spec(temp)
        segmentor.release()
        return temp

    def tag(self,wordlist):
        postagger = Postagger()
        postagger.load(self.pos_model_path)
        postag_list = list(postagger.postag(wordlist))
        return postag_list

    def ner(self,wordlist, postag_list):
        recognizer = NamedEntityRecognizer()
        recognizer.load(self.ner_model_path)
        netag_list = list(recognizer.recognize(wordlist, postag_list))
        return netag_list

    def parser(self,words, postags):
        parser = pyltp.Parser()
        parser.load(self.par_model_path)
        arcs = parser.parse(words, postags)
        # print('\t'.join('%d:%s' % (arc.head, arc.relation) for arc in arcs))
        parser.release()
        return arcs

    #找到主语的index
    def get_sbv_id(self,netags, arcs, key_word_index):
        n = 0
        for i in arcs:
            # 这三个名词词性可以代表人名，机构名，我们
            is_main_perosn = 'nh' in netags[n] or 'pronoun' in netags[n] or 'ni' in netags[n]
            if is_main_perosn or i.head == key_word_index+1 or i.relation =='SBV':
                #这里根据中心词的index找到主语；其中i.head == key_word_index+1，不容易理解，这里举一个例子，我想听歌，分词结果是 我 想 听歌
                #注意这里中心词的index是相对分词结果的list而言的，所以中心词想的index是1，通过句子依存分析发现我这个字的head刚好是2，所以就对上了
                #搞不清楚了自己跑一下分析结果就理解了
                return n#这里返回的n仍然是相对分词结果的list的index
            n +=1
        return None

    #找到关键词的索引
    def get_key_words_index(self,wordslist):
        for i,word in enumerate(wordslist):
            if word in self.key_words:
                return i
        return None

    # 寻找依存树根节点编号,相比上面一种从Wordlist方式找关键词索引，这是第二种方法
    def get_dependtree_root_index(self, word_list):
        # 词性标注
        postags = self.tag(word_list)
        # print(list(postags))

        # 命名实体识别
        netags = self.ner(word_list, postags)
        # print(list(netags))

        # 句法依存关系
        arcs = self.parser(word_list, postags)
        # print(' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

        for i in range(len(arcs)):
            if arcs[i].head == 0:
                # 如果arcs[i].head==0 则表示找到了整个一句话的核心词，那么返回核心词的索引
                return i, postags, arcs  # 同时返回词性及依存关系列表
        return -1, postags, arcs

    #这个项目关于找到言论内容有两个不错值得借鉴的方法，第一个是根据标点符号在提取言论之前就将文本分为一个个的句子，然后以句子为单位再去提取，因此
    #句子的分隔符自然是以。 ？ ！来切分的；第二种方法是在两个关键词之间去考察，也就是说，一个关键词后面如果有多个句子，那么我们根据后面的句子与挨着关键词
    #最近的句子之间的相似度来判断后面的句子是否和前面的言论一样，这样应该会科学一点，这个方法主要使用了普林斯顿大学的那篇论文
    def get_target_sentence(self,text):
        text = text.replace('\n','')
        sentences = re.split('(。|！|\!|\.|？|\?)',text)
        new_sents = []
        for i in range(int(len(sentences)/2)):
            sent = sentences[2*i] + sentences[2*i+1]
            #这里比较精妙，对于非科班出身来说，不容易想到这样子去提取，看不懂就自己写个
            #句子测试，奇数位刚好均为标点符号
            new_sents.append(sent)
        target_sentences = []
        for sent in target_sentences:
            words = list(jieba.cut(sent))
            if len([word for word in words if word in self.key_words])>0 and len(words)>5:
                #这里认为包含关键词并且一个句子至少含有5个词，我们才认为属于目标句子
                target_sentences.append(sent)
        return target_sentences
    #这里根据上面一个算法，将内容视为关键词后面的所有内容提取即可
    def find_content(self,word_list, key_word_index):
        content = []
        for i in range(key_word_index + 1, len(word_list)):
            content.append(word_list[i])
        return ''.join(content)
####################################################
    #下面开始写第二种获取言论内容的方法
    # 获取中心词之后的第一个符号（，。！）的索引
    def get_first_wp_after_keyword(self, word_list, key_word_index):
        for i in range(key_word_index + 1, len(word_list)):
            if word_list[i] == '.' or word_list[i] =='。' or word_list[i] == '!' or word_list[i] == '?' or word_list[i] == '？':
                #在词性标注里面wp一般指的是，。！这样的符号
                return i
        return 0

    # 获取句号索引列表
    def get_periods_index_after_keyword(self, word_list, index):
        #这里的index最好以中心词之后的第一个分句标点符号为准
        ret = []
        for i in range(index + 1, len(word_list)):
            if word_list[i] in ['。', '？', '！']:
                ret.append(i)
        return ret

    #获取一个关键字后面的所有句子序列，为后续句子相关性分析做准备
    def keyword_senteces(self,word_list,postags,key_word_index):
        sentences = []
        first_wp_index = self.get_first_wp_after_keyword(word_list,key_word_index)
        # print('first_wp_index:',first_wp_index)
        first_sentence = ''.join(word_list[key_word_index+1:first_wp_index])#得到关键词后面的第一个句子
        # print('first_sentence:',first_sentence)
        sentences.append(first_sentence)
        other_related_sentence_wp_index_list = self.get_periods_index_after_keyword(word_list,first_wp_index)
        # print('other_related_sentence_wp_index_list:',other_related_sentence_wp_index_list)
        #以关键字之后第一个句子后面的标点符号索引找到后面的句子，并根据标点符号切分
        if len(other_related_sentence_wp_index_list) > 0:#如果找到了
            for i ,index in enumerate(other_related_sentence_wp_index_list):#一个个的截取
                if i ==0:
                    sentences.append(''.join(word_list[first_wp_index:index]))#当为第一个句子的时候，把它的索引和第一个句子结尾的索引截取就得到了第一个相关句子
                else:
                    sentences.append(''.join(word_list[other_related_sentence_wp_index_list[i-1]:index]))#第二个句子就是当前索引与上一个索引截取就对了
        return sentences

import numpy as np
from collections import Counter

class sentence_embedding():
    def __init__(self,model,corpus_file,a = 0.001):
        self.model = model
        self.a = a
        self.corpus_file = corpus_file

    def word_freq(self):
        word_list = []
        with open(self.corpus_file, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                word_list += line.split()
        cc = Counter(word_list)
        num_all = sum(cc.values())
        def get_word_freq(word):
            return cc[word] / num_all
        return get_word_freq

    # 获得句子向量矩阵
    def get_sentences_vec(self, sent_list, get_wd_freq):
        # 词向量加权部分
        row = self.model.wv.vector_size  # 获取在Word2vector模型中每个词的维度是多少
        col = len(sent_list)  # 获得一共有多少个句子需要参加比较
        sent_mat = np.mat(np.zeros((row, col)))  # 将np.zeros((row, col))这样一个二维数组转为矩阵
        for i, sent in enumerate(sent_list):
            # new_sent = rm_spec(sent)
            new_sent = sent
            if not new_sent: continue
            sent_vec = np.zeros(row)
            for word in new_sent:
                pw = get_wd_freq(word)
                w = self.a / (self.a + pw)
                try:
                    vec = np.array(self.model.wv[word])
                    sent_vec += w * vec  # 这里相当于将每个词的向量经过与自己的权重相乘之后叠加在一起，叠加在一起之后还是一个1*row的向量
                except:
                    pass
            sent_mat[:, i] += np.mat(sent_vec).T  # sent_vec是一行的向量，根据sent_mat = np.mat(np.zeros((row, col)))的定义
            # 需要将每次句子表示成竖着的列向量，所以做了一次转置
            sent_mat[:, i] /= len(new_sent)  # 根据论文，这里需要除以整个句子的长度

        # 减去PCA中的第一主成分
        u, s, vh = np.linalg.svd(sent_mat)  # 这里类似线性代数里面的矩阵相似一样，中间那个矩阵是特征值组成的
        # 只不过这个函数可以自动将为0的特征值去掉缩小维度，参考博客：https://blog.csdn.net/u012162613/article/details/42214205
        sent_mat = sent_mat - u * u.T * sent_mat
        # 论文上面是这样相减，但是具体原因还没有明白https://www.ctolib.com/topics-132206.html
        return sent_mat

    # 计算余弦相似度
    def get_cos_similarity(self,v1, v2):
        if len(v1) != len(v2):
            return 0
        return np.sum(np.array(v1) * np.array(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # 返回句子向量矩阵中各列向量与第一列向量的相似度
    def get_sent_vec_sims(self,sent_mat):
        first = np.array(sent_mat[:, 0])[:, 0]
        col = sent_mat.shape[1]
        sims = [1.0]
        if col > 1:
            for i in range(1, col):
                vec = np.array(sent_mat[:, i])[:, 0]
                sims.append(self.get_cos_similarity(first, vec))
        return sims

    # 获得最终说的话
    def get_final_sents(self,sents):
        content = []
        first_sentenc = sents[0]
        content.append(first_sentenc)
        get_word_frequency = self.word_freq()
        sents_vector = self.get_sentences_vec(sents,get_word_frequency)
        sims = self.get_sent_vec_sims(sents_vector)
        threshold = 0.5  # 相似度超过0.5即认为两句话相关
        for i in range(len(sims)):
            if sims[i] > threshold:
                content.append(sents[i])
        ret = [''.join(s) for s in content]
        return ret




























