import numpy as np
import pickle
import csv
import sys
import random


taskfile = open('task2.txt', 'r')
tasks = csv.reader(taskfile, delimiter=' ')
testing = []
for line in tasks:
    if line[0] != ':':
        testing.append(line)

file = open(sys.argv[1], 'rb')
data = pickle.load(file)
luck = ['center', 'context']
w = data[0][random.choice(luck)]
vocab_dict = data[1][0]
int_dict = data[1][1]

def find_nearest(center, c_embed, k, flag):
    scores = np.matmul(c_embed, np.transpose(center))
    ind = np.argsort(flag*scores)
    similar = []
    for i in range(k):
        similar.append(int_dict[ind[i]])
        
    return similar

correct = 0
total = 0
for test in testing:
    try:
        fir = vocab_dict[test[0].lower()]
        sec = vocab_dict[test[1].lower()]
        thr = vocab_dict[test[2].lower()]
        predict_vector = w[sec,:] - w[fir,:] + w[thr,:]
        nearest = find_nearest(w,predict_vector,random.randint(10,40),-1)
        if test[3].lower() in nearest:
            correct += 1
        total += 1
    except:
        pass
    
accuracy = correct/total
print('Accuracy = ' + str(accuracy))
