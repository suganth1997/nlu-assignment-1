#
#
# python negative_sampling.py [window_size] [wordvec_dimension] [epochs] [num_negative_samples]
#
#
import tensorflow as tf
import numpy as np
import nltk
import sys
from nltk.corpus import reuters, stopwords
import pickle
import re
from random import randint, uniform
import math

sentences = reuters.sents()

sentences = list(sentences)

stop = stopwords.words()

def find_sublist(sub, l):
    sub_in = []
    for i in range(len(l) - len(sub) + 1):
        if l[i:i+len(sub)] == sub:
            sub_in.append(i)
            
    return sub_in


subs_dict = {'usa':['u','s'], 'uk ':['u','k']}

for i,sentence in enumerate(sentences):

    new_sent = []

    for word in sentence:

        if len(re.findall('[a-zA-Z]', word)) != 0:

            new_sent.append(word)

    new_sent = [i.lower() for i in new_sent]

    for v,k in subs_dict.items():

        sub = find_sublist(k,new_sent)

        sub.reverse()

        for index in sub:

            new_sent[index] = v

            del new_sent[index + 1]

    new_sent = [word for word in new_sent if len(word) > 2]       

    sent = [i for i in new_sent if i.lower() not in stop]

    sentences[i] = sent
    
window_size = int(sys.argv[1]) #Only odd and prolly not 1

flat_words = [x for sentence in sentences for x in sentence]

vocab = list(set(flat_words))

vocab_dict = dict(zip(vocab,range(len(vocab))))
int_dict = dict(zip(range(len(vocab)),vocab))

#Helper function
def get_window_indices(window_size):
    window = []
    i = 1
    while len(window) != window_size - 1:
        window += [i, -i]
        i += 1
        
    return window
        
print(get_window_indices(window_size))

Y = []
X = []
win_index = get_window_indices(window_size)

for sentence in sentences:
    for i,word in enumerate(sentence):
        num_times = 0
        for j in win_index:
            if i - j < 0 or i - j >= len(sentence):
                continue
            else:
                num_times += 1
                Y.append(vocab_dict[sentence[i-j].lower()])
                
        X += [vocab_dict[word.lower()]]*num_times
		
validation = 50000
x_valid = X[-validation:]
y_valid = Y[-validation:]
X = X[0:-validation]
Y = Y[0:-validation]

embedding_len = int(sys.argv[2])

def find_nearest(center, context, word, k):
    c_embed = center[vocab_dict[word.lower()],:]
    scores = np.matmul(context, c_embed)
    ind = np.argsort(-scores)
    similar = []
    for i in range(k):
        similar.append(int_dict[ind[i]])
        
    return similar


#Negative Sampling
k = int(sys.argv[4])
set_size = len(X)
batch_size = 1
epochs = int(sys.argv[3])
x = tf.placeholder(tf.int64)
y = tf.placeholder(tf.int64)
negative_samples = tf.placeholder(tf.int64)
embeddings = {
    'center':tf.Variable(tf.random_normal([len(vocab), embedding_len], dtype = tf.float64)), 
    'context':tf.Variable(tf.random_normal([len(vocab), embedding_len], dtype = tf.float64))
}

c_embed = tf.nn.embedding_lookup(embeddings['center'], x)
ct_embed = tf.nn.embedding_lookup(embeddings['context'], y)
positive_dot = tf.log(tf.sigmoid(tf.tensordot(c_embed, ct_embed, 1)))
negatives = tf.nn.embedding_lookup(embeddings['context'], negative_samples)
negative_dot = tf.reduce_sum(tf.log(
    tf.sigmoid(-tf.matmul(negatives, tf.reshape(c_embed, [embedding_len, 1])))
    ))
loss = - positive_dot - negative_dot
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        a = 0
        while a < set_size:
            x_train = X[a]
            y_train = Y[a]
            for i in range(10):
                [t, loss_] = sess.run([train, loss], feed_dict = {x:x_train, y:y_train, 
                                                negative_samples:np.random.choice(len(vocab), k, replace=False)})
                if i==9:
                    print(loss_)
            if a%1000 == 0:
                wordvec = sess.run(embeddings)
                print('Epoch = ' + str(epoch + 1) + ', window_size = ' + str(window_size) + ', wordvec_dim = ' + str(embedding_len) + ', num_neg_samples = ' + str(k))
                print('Loss = ' + str(loss_) + ', Observation_index = ' + str(a) + '/' + str(set_size))
                target_word = int_dict[np.random.randint(0,len(vocab))]
                print('Nearest words to ' + target_word + ' are ' + str(find_nearest(wordvec['center'], wordvec['context'], 
                                                                target_word, 4)))
                print('Nearest words to government are ' + str(find_nearest(wordvec['center'], wordvec['context'], 'government', 4)))
                print('')
            a = a + 1
            
        if epoch%2 != 0:
            filename = 'negative_sampling_' + 'window_size_' + str(window_size) + '_wordvec_dim_' + str(embedding_len) + '_epochs_' + str(epoch + 1) + '_num_neg_samples_' + str(k) + '.dat'
            file = open(filename, 'wb')
            wordvec = sess.run(embeddings)    
            pickle.dump([wordvec, [vocab_dict, int_dict], [x_valid, y_valid]], file)
            file.close()
            
    








