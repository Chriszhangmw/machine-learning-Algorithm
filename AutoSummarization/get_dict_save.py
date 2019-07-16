import pickle
from collections import Counter
import  numpy as np



tokeners = np.load('./tokeners.npy')
def get_counts(tokeners):
    tokener_counter = Counter(tokeners)

    total_cnt = sum(tokener_counter.values())

    frequence = {w: count/total_cnt for w,count in tokener_counter.items()}
    with open('./frequence.pkl','wb') as f:
        pickle.dump(frequence, f, pickle.HIGHEST_PROTOCOL)
    return frequence

# a = get_counts(tokeners)


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj():
    with open('./frequence.pkl', 'rb') as f:
        return pickle.load(f)

a = load_obj()
print(len(a))
print(max(a.values()))

# occurences_frequences = sorted(a.items(),key = lambda x:x[1],reverse=True)[:10000]
# print(len(occurences_frequences))
# print(occurences_frequences)

