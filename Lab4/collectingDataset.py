import pickle

import numpy as np
from keras.layers import Embedding
import ssl
from keras.datasets import imdb
from keras import preprocessing
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense
from sklearn.metrics.pairwise import cosine_similarity

glove_dir = '/Users/simonakesson/PycharmProjects/EDAN95Assignment1/EDANlab4/'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

what = embeddings_index['sweden']
print('Found %s word vectors.' % len(embeddings_index))
cos_lib_table = list()
cos_lib_france = list()
cos_lib_sweden = list()
i = 1
words = []
ner = []
for row in embeddings_index:
    a = embeddings_index[row].reshape(1, 100)
    b = embeddings_index['table'].reshape(1, 100)
    c = embeddings_index['france'].reshape(1, 100)
    d = embeddings_index['sweden'].reshape(1, 100)
    cos_lib_table.append((row, cosine_similarity(a, b)[0][0]))
    cos_lib_france.append((row, cosine_similarity(a, c)[0][0]))
    cos_lib_sweden.append((row, cosine_similarity(a, d)[0][0]))
    dit = cos_lib_france[0][1]
    words.append(row.lower())
    i += 1
    print(i)


pickle.dump(words, open("words_from_gloVe.p", "wb"))
top_table = sorted(cos_lib_table, key=lambda x: x[1], reverse=True)
top_france = sorted(cos_lib_france, key=lambda x: x[1], reverse=True)
top_sweden = sorted(cos_lib_sweden, key=lambda x: x[1], reverse=True)

print('table', top_table[1:6])
print('france', top_france[1:6])
print('sweden', top_sweden[1:6])


print()
