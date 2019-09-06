import numpy as np

#itial the dictionary
tag2id,id2tag = {},{}
word2id,id2word = {},{}

#build the dictionary
for line in open('./traindata.txt'):
    line = line.split('/')
    word,tag = line[0],line[1].rstrip()
    if word not in word2id:
         word2id[word] = len(word2id)
         id2word[len(word2id)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(tag2id)] = tag

# initial some parameters
M = len(word2id) #the number of X
N = len(tag2id) #the number of Z

pi = np.zeros(N)
B = np.zeros((N,M)) #estim
A = np.zeros((N,N)) #trans

#calculate
prev_tag = ''
for line in open('traindata.txt'):
    line = line.split('/')
    word_id,tag_id = word2id[line[0]],tag2id[line[1].rstrip()]
    if prev_tag == '':
        pi[tag_id] +=1
        B[tag_id][word_id] +=1
    else:
        A[tag2id[prev_tag]][tag_id] +=1
        B[tag_id][word_id] +=1

    if line[0] == '.':#end of the sentence
        prev_tag = ''
    else:
        prev_tag = line[1].rstrip()

#translate to the probability
def log(v):
    if v == 0:
        return np.log(v + 0.000001) #smoothing
    return np.log(v)

# viterbi algorithm
def viterbi(x,pi,A,B):
    x = [word2id[word] for word in x.split(' ')]
    T = len(x)

    dp = np.zeros((T,N))
    ptr = np.zeros((T,N),dtype=int)

    for j in range(N):
        dp[0][j] = log(pi[j]) + log(B[j][x[0]])

    for i in range(1,T):
        for j in range(N):
            dp[i][j] = -9999
            for k in range(N):
                score = dp[i-1][k] + log(A[k][j]) + log(B[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k #k represents the previous word state index

    best_seq = [0]*T
    best_seq[T-1] = np.argmax(dp[T-1])
    for i in range(T-2,-1,-1):
        best_seq[i] = ptr[i+1][best_seq[i+1]] # 等式后面是表示第I个词再第J个状态下的，ptr中保存的恰好是I个词J个状态下的前一个词的最佳状态编号，所以赋值给i(从i+1)
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])

if __name__ == '__main__':
    x = 'keep positive to life'
    viterbi(x,pi,A,B)


