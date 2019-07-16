from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import pandas as pd
import numpy as np
import jieba
from gensim.models.word2vec import LineSentence
from functools import reduce
import  networkx
import re


# model = FastText(LineSentence('./corpus_cutted.txt'), window=5, size=35, iter=10, min_count=1)
# model.save('./news.model')




model =FastText.load('./news.model')
print(model['小米'])






