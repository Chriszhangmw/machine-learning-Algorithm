

from gensim.models import Word2Vec
from collections import defaultdict



def get_model_from_file(filename):
    model = Word2Vec.load(filename)
    return model
model = get_model_from_file('./final_model')

def get_related_word(inial_words,model):
    max_num = 50
    seen = defaultdict(int)
    need_seen = [inial_words]
    while need_seen and len(seen) < max_num:
        node = need_seen.pop(0)
        new_expanding = [w for w,s in model.most_similar(node,topn = 20)]
        need_seen += new_expanding
        seen[node] +=1
    del seen[inial_words]
    seen = sorted(seen.items(),key=lambda kv: kv[1], reverse=True)
    return  seen

print(model.most_similar('说',topn = 20))














