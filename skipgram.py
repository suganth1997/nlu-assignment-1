#
# Run as 
# 
# python skipgram.py [window_size] [embedding_len] [epochs]  
#
#
import tensorflow as tf
import numpy as np
import nltk
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import reuters, stopwords
import pickle
import re
import sys
from random import randint

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

def get_window_indices(window_size):
    window = []
    i = 1
    while len(window) != window_size - 1:
        window += [i, -i]
        i += 1
        
    return window


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

#Tensorflow begins

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

embeddings = {
    'centre':tf.Variable(tf.random_normal([len(vocab), embedding_len], dtype = tf.float64)), 
    'context':tf.Variable(tf.random_normal([embedding_len, len(vocab)], dtype = tf.float64))
}

def neural_net(x):
    c_embed = tf.nn.embedding_lookup(embeddings['centre'],x) #tf.matmul(tf.one_hot(x, len(vocab)), embeddings['centre'])
    out = tf.matmul(c_embed, embeddings['context'])
    return tf.nn.softmax(out)

init = tf.global_variables_initializer()

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_nearest(center, context, word, k):
    c_embed = center[vocab_dict[word],:]
    scores = np.matmul(c_embed, context)
    ind = np.argsort(-scores)
    similar = []
    for i in range(k):
        similar.append(int_dict[ind[i]])
        
    return similar


logits = neural_net(x)
prob = tf.cast(tf.one_hot(y,len(vocab)), dtype = tf.float64)*(logits)
loss_op = tf.reduce_mean(-tf.log(tf.reduce_sum(prob, 1)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss_op)

set_size = len(X)
batch_size = 1
epochs = int(sys.argv[3])


init = tf.global_variables_initializer()
all_loss = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        a = 0
        while a < set_size:
            dummy_b = a + batch_size
            b = dummy_b if dummy_b < set_size else set_size
            x_train = X[a:b]
            y_train = Y[a:b]
            a = a + batch_size
            
            [t, loss] = sess.run([train_op, loss_op], feed_dict = {x:x_train,y:y_train})
            all_loss.append(loss)
            if a%1000 == 0:
                print('Window size = ' + str(window_size) + ', Wordvec dimension = ' + str(embedding_len))
                print('Epoch = ' + str(epoch + 1) + ', Observation number ' + str(a - 1000*batch_size) + ' to ' + str(b), end = ' ')
                print('Loss = ' + str(loss) + ', Observation_index = ' + str(a) + '/' + str(set_size))
                wordvec = sess.run(embeddings)
                target_word = int_dict[np.random.randint(0,len(vocab))]
                print('Nearest words to ' + target_word + ' are ' + str(find_nearest(wordvec['centre'], wordvec['context'], 
                                                                target_word, 4)))
                print('Nearest words to government are ' + str(find_nearest(wordvec['centre'], wordvec['context'], 
                                                                'government', 4)))
                print('')
                
        if epoch%2 == 1:
            wordvec = sess.run(embeddings)
            filename = 'skipgram_window_' + str(window_size) + '_wordvecdim_' + str(embedding_len) + '_epoch_' + str(epoch) + '.dat'
            file = open(filename, 'wb')
            pickle.dump([wordvec, all_loss, [vocab_dict, int_dict], [x_valid, y_valid]], file)
            file.close()
