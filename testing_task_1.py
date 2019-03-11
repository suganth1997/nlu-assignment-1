import tensorflow as tf
import numpy as np
import pickle
import os
import csv
import scipy.stats
import sys

def calc_valid_prob(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    embeddings = data[0]
    vocab_dict = data[1][0]
    int_dict = data[1][1]
    x_valid = data[2][0]
    y_valid = data[2][1]
    center_embed = tf.constant(embeddings['center'], dtype = tf.float64)
    context_embed = tf.constant(embeddings['context'], dtype = tf.float64)
    init = tf.global_variables_initializer()
    dot_prod = tf.reduce_sum(
            tf.nn.embedding_lookup(center_embed, x_valid)*tf.nn.embedding_lookup(context_embed, y_valid)
        ,1)
    with tf.Session() as sess:
        sess.run(init)
        prob = sess.run(tf.reduce_mean(tf.sigmoid(dot_prod)))
    return prob

def calc_simlex_rating(dataset, embedding_pickle):
    file = open(dataset, 'r')
    sim = csv.reader(file, delimiter='\t')
    simlex = []
    for line in sim:
        break   
    for line in sim:
        simlex.append([line[0],line[1],float(line[3])])
        
    embed = open(embedding_pickle, 'rb')
    data = pickle.load(embed)
    w = data[0]
    vocab_dict = data[1][0]
    cosine_sim = []
    simlex_rating = []
    for rating in simlex:
        try:
            x = w['center'][vocab_dict[rating[0]]]
            y = w['context'][vocab_dict[rating[1]]]
            cosine_sim.append(np.dot(x/np.linalg.norm(x),y/np.linalg.norm(y)))
            simlex_rating.append(rating[2])
        except:
            pass
        
    return list(scipy.stats.spearmanr(cosine_sim,simlex_rating))[0]

direc = sys.argv[1]
simlex_direc = 'simlex.txt'
for filename in os.listdir(direc):
    if filename.endswith('.dat'):
        print(filename)
        print('Spearman Correlation = ', end = '')
        print(calc_simlex_rating(simlex_direc, direc + filename))
        print('')
