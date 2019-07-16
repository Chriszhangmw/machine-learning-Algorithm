import re

import jieba
import pandas as pd



def get_stop_words(stop_words_file):
    ret = [s.strip() for s in open(stop_words_file, 'r', encoding='utf-8').readlines()]
    return ret


stop_words = get_stop_words('stop_words.txt')


def sentence_preprocess(sent):
    sent = sent.replace('\n', ' ')
    words = jieba.lcut(sent)
    # words = pkuseg.pkuseg().cut(sent) # much slower than jieba
    ret = []
    for w in words:
        if w not in stop_words:
            ret.append(w)
    ret = ' '.join(ret)
    ret = re.sub('\s\s+', ' ', ret)
    return ret


def data_process(outfile, data):
    with open(outfile, 'w', encoding='utf-8') as fout:
        fout.write(','.join(data.columns))
        fout.write('\n')
        for i in range(len(data)):
            if i % 500 == 0:
                print(i)
            sent = data['content'][i]
            words = sentence_preprocess(sent)
            if words:
                fout.write('{},"""{}"""'.format(data.iloc[i, 0], words))
                for j in range(2, 22):
                    fout.write(',{}'.format(data.iloc[i, j]))
                fout.write('\n')


train_data = pd.read_csv(
    'D:\\Eclipse_workplace\\Training\\NLPTraining\\project3\\data\\ai_challenger_sentiment_analysis_trainingset_20180816\\sentiment_analysis_trainingset.csv')

validation_data = pd.read_csv(
    'D:\\Eclipse_workplace\\Training\\NLPTraining\\project3\\data\\ai_challenger_sentiment_analysis_validationset_20180816\\sentiment_analysis_validationset.csv')

test_data = pd.read_csv(
    'D:\\Eclipse_workplace\\Training\\NLPTraining\\project3\\data\\ai_challenger_sentiment_analysis_testa_20180816\\sentiment_analysis_testa.csv')

data_process('trainset.csv', train_data)
data_process('validationset.csv', validation_data)
data_process('testset.csv', test_data)
