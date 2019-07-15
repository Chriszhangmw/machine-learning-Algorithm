import os

from pyltp import NamedEntityRecognizer  # 命名实体识别
from pyltp import Parser  # 句法解析
from pyltp import Postagger  # 词性标注
from sklearn.feature_extraction.text import TfidfVectorizer

LTP_DATA_DIR = 'E:\\PythonProjects\\Training\\NLPTraining\\project01\\model\\ltp-models'


class MyLtp:

    def __init__(self):
        self.postagger = Postagger()
        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
        self.postagger.load(pos_model_path)

        self.recognizer = NamedEntityRecognizer()
        ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
        self.recognizer.load(ner_model_path)

        self.parser = Parser()
        par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
        self.parser.load(par_model_path)

    def clean(self):
        self.postagger.release()
        self.recognizer.release()
        self.parser.release()

    # 寻找依存树根节点编号
    def get_dependtree_root_index(self, word_list):
        # 词性标注
        postags = self.postagger.postag(word_list)
        # print(list(postags))

        # 命名实体识别
        netags = self.recognizer.recognize(word_list, postags)
        # print(list(netags))

        # 句法依存关系
        arcs = self.parser.parse(word_list, postags)
        # print(' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

        for i in range(len(arcs)):
            if arcs[i].head == 0:
                return i, postags, arcs  # 同时返回词性及依存关系列表
        return -1, postags, arcs

    # 寻找依存关系子节点
    def get_child_index(self, ind, arcs):
        ret = []
        for i in range(len(arcs)):
            if arcs[i].head == ind + 1:
                ret.append(i)

        return ret

    # 获取命名实体索引
    def get_ne_index(self, postags, chd_list):
        ret = []
        for i in chd_list:
            if postags[i] in ['n', 'nh', 'ni']:
                ret.append(i)
        return ret

    # 获取中心词之后的第一个符号的索引
    def get_first_wp_after_index(self, postags, after):
        for i in range(after + 1, len(postags)):
            if postags[i] == 'wp':
                return i
        return 0

    # 获取句号索引列表
    def get_periods_index_after(self, word_list, after):
        ret = []
        for i in range(after + 1, len(word_list)):
            if word_list[i] in ['。', '？', '！']:
                ret.append(i)
        return ret

    # 获取长句中的分句，为下面的句子向量分析作准备
    def get_sent_list(self, word_list, start, periods):
        ret = []
        if len(periods) == 0:
            ret.append(list(word_list[start + 1:]))
        for i, p in enumerate(periods):
            if i == 0:
                ret.append(list(word_list[start + 1:p + 1]))
            else:
                ret.append(list(word_list[periods[i - 1] + 1:p + 1]))
        return ret

    # # 获取语料库TF-IDF vectorizer
    # def get_tfidf_vectorizer(self, corpus_file):
    #     corpus = []
    #     with open(corpus_file, 'r', encoding='utf-8') as f:
    #         while True:
    #             line = f.readline()
    #             l = line.strip()
    #             if l:
    #                 corpus.append(l)
    #             else:
    #                 break
    #
    #     vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')  # 不过滤单汉字
    #     X = vectorizer.fit_transform(corpus)
    #     return vectorizer

    # # 获取句子向量
    # def get_sentence_vec(self, vectorizer, word_list):
    #     trans = vectorizer.transform([' '.join(word_list)])
    #     return trans.toarray()[0]

    # words: 要识别的内容词语列表, talk_sims: “说”的近义词
    def get_character_speech(self, words, talk_sims):
        # 获取中心词，词性列表，依存关系表
        root_index, postags, arcs = self.get_dependtree_root_index(words)
        # print('index:', root_index)
        # print('len words:', len(words))
        # print('root:', words[root_index])

        # 中心词不在近义词列表，返回空值
        if words[root_index] not in talk_sims:
            return '', '', []

        wp_index = self.get_first_wp_after_index(postags, root_index)
        if wp_index == 0: wp_index = root_index
        # print('wp_index:', wp_index)

        sent_split_idx = self.get_periods_index_after(words, wp_index)
        # print('split:', sent_split_idx)

        # 分句
        sents = self.get_sent_list(words, wp_index, sent_split_idx)
        # print('sents: ', sents)
        # for sen in sents:
        #     print('sen: ', sen)

        # 获取完整命名实体，针对命名实体词被分割的情况
        children = self.get_child_index(root_index, arcs)
        # print(children)

        ne_list = self.get_ne_index(postags, children)

        oth = []
        for ne in ne_list:
            nechd = self.get_child_index(ne, arcs)
            oth.append(self.get_ne_index(postags, nechd))

        # print('ne_list: ', ne_list)
        # print('oth: ', oth)

        if ne_list:
            for i, n in enumerate(ne_list):
                if oth[i]:
                    ne = words[oth[i][0]] + words[n]
                    # print(words[oth[i][0]] + words[n])
                else:
                    ne = words[n]
                    # print(words[n])
                return ne, words[root_index], sents
        else:
            return '', '', []
