import numpy as np
import torch
from torch import nn, optim
import random
from collections import Counter
import matplotlib.pyplot as plt

#training data
text = 'KTH Royal Institute of Technology is the largest and most ' \
       'respected technical university in Sweden—ranked top 100 in the ' \
       '2020 QS World University Rankings. By choosing KTH, you gain access' \
       ' to a vibrant student life and a prestigious academic environment'

#parameters
embedding_dim = 2 #the dim of the vector
print_every = 1000
epochs = 1000
batch_size = 5
N_samples = 3
window_size = 5
freq = 0
deleter_words = False

#text processing
def preprocess(text,freq):
    text = text.lower()
    words = text.split()
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]
    return  trimmed_words

words = preprocess(text,freq)

#build our vocabulary
vocabulary = set(words)
vocabulary2index = {w:c for c,w in enumerate(vocabulary)}
index2vocabulary = {c:w for c,w in enumerate(vocabulary)}

#trans text into numerical
index_words = [vocabulary2index[w] for w in words]

#calculate the frequency of each word
word_count = Counter(index_words)
total_count = len(word_count)
word_freqs = {w:c/total_count for w,c in word_count.items()}

#delete some high frequency words
if deleter_words:
    t = 1e-5
    prob_drop = {w:1-np.sqrt(t/word_freqs[w]) for w in index_words}
    train_words = {w for w in index_words if random.random() < (1-prob_drop[w])}
else:
    train_words = index_words

#distribution of words
word_freqs = np.array(list(word_freqs.values()))
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

#get the target words
def get_target(words,index,window_size):
    target_window = np.random.randint(1,window_size+1)
    start_point = index-target_window if (index-target_window) > 0 else 0
    end_point = index + target_window
    targets = set(words[start_point:index] + words[index +1:end_point + 1])
    return list(targets)

# yield batch
def get_batch(words,batch_size,window_size):
    n_batches = len(words)//batch_size
    words = words[:n_batches*batch_size]
    for idx in range(0,len(words),batch_size):
        batch_x, batch_y = [],[]
        batch = words[idx:idx+batch_size]
        for i in range(len(batch)):
            x = batch[i]
            y = get_target(batch,i,window_size)
            batch_x.extend([x]*len(y))
            batch_y.extend(y)
        yield batch_x,batch_y

#define the model
class SkipGramNeg(nn.Module):
    def __init__(self,n_vocab,n_embed,noise_dist):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist

        self.in_embed = nn.Embedding(n_vocab,n_embed)
        self.out_embed = nn.Embedding(n_vocab,n_embed)

        self.in_embed.weight.data.uniform_(-1,1)
        self.out_embed.weight.data.uniform_(-1,1)

    #forward process
    def forward_input(self,input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors
    #target word forward
    def forward_output(self,output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    #negative words forward
    def forward_noise(self,size,N_sample):
        noise_dist = self.noise_dist
        noise_words = torch.multinomial(noise_dist,
                                        size * N_sample,
                                        replacement=True)
        noise_vectors = self.out_embed(noise_words).view(size, N_sample, self.n_embed)
        return noise_vectors
#define the loss function
class NegativeSampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input_vectors,output_vectors,noise_vectors):
        batch_size,emded_size = input_vectors.shape
        input_vectors = input_vectors.view(batch_size,emded_size,1)
        output_vectors = output_vectors.view(batch_size,1, emded_size)# can be multiply with each other

        #loss value of target word
        out_loss = torch.bmm(output_vectors,input_vectors).sigmoid().log()
        #the size is [batchsize,1,1], so we need to reduce the dim
        out_loss = out_loss.squeeze()

        #loss value of negative word
        noise_loss = torch.bmm(noise_vectors.neg(),input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)

        return -(out_loss + noise_loss).mean()

model = SkipGramNeg(len(vocabulary2index),embedding_dim,noise_dist)
criterion = NegativeSampleLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

#training
steps = 0
for e in range(epochs):
    #get the inputs word and target word
    for input_words,target_word in get_batch(train_words,batch_size,window_size):
        steps +=1
        inputs,targets = torch.LongTensor(input_words),torch.LongTensor(target_word)
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        size,_ = input_vectors.shape
        noise_vectors = model.forward_noise(size,N_samples)

        loss = criterion(input_vectors,output_vectors,noise_vectors)
        if steps%print_every ==0:
            print('loss: ',loss)
        #graident backword
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for i,w in index2vocabulary.items():
    vectors = model.state_dict()['in_embed.weight']
    x,y = float(vectors[i][0]),float(vectors[i][1])
    plt.scatter(x,y)
    plt.annotate(w,xy = (x,y),xytext=(5,2),textcoords = 'offset points', ha='right', va='bottom')
plt.show()








































