import csv
import numpy as np
import matplotlib.pyplot as plt
correlation_file = open('correlation_scores.csv', 'r')
corr = csv.reader(correlation_file, delimiter=',')
values = []
for line in corr:
    break
for line in corr:
    values.append([float(v) for i,v in enumerate(line)])
    #print([float(i) for i in line])

values = np.array(values)
spearman_corr = values[:,5]
valid_prob = values[:,4]
labels = values[:,0:4]

ticks = [', '.join([str(i) for i in labels.astype(int)[j]]) for j in range(len(labels))]

fig, ax = plt.subplots()
ax.bar(range(len(values)),spearman_corr, tick_label = ticks)
ax.set_xticklabels(ticks,rotation = 'vertical',fontsize = 10)
ax.set_xlabel('Win_size, Embed_size, Batch_size, Num_neg_samples', fontsize = 15)
ax.set_ylabel('Spearman Correlation', fontsize = 15)
ax.set_title('Model Performance',fontsize = 25)
plt.tight_layout()

fig1, ax1 = plt.subplots()
ax1.bar(range(len(values)),valid_prob, tick_label = ticks)
ax1.set_xticklabels(ticks,rotation = 'vertical',fontsize = 10)
ax1.set_xlabel('Win_size, Embed_size, Batch_size, Num_neg_samples', fontsize = 15)
ax1.set_ylabel('Validation Probability', fontsize = 15)
ax1.set_title('Model Performance',fontsize = 25)
plt.tight_layout()
plt.show()
