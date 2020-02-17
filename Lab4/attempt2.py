import pickle
import sys
import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, SimpleRNN, Dropout, Bidirectional
from keras.utils import to_categorical
from sklearn.feature_extraction import DictVectorizer
import time
from keras import models, layers
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import math
from keras.preprocessing.sequence import pad_sequences

from EDANlab4.conll_dictorizer import Token

OPTIMIZER = 'adam'
BATCH_SIZE = 64
EPOCHS = 3
MINI_CORPUS = False
W_SIZE = 2
EMBEDDING_DIM = 100
MAX_LEN = 150

def save(file, corpus_dict, column_names):
    """
    Saves the corpus in a file
    :param file:
    :param corpus_dict:
    :param column_names:
    :return:
    """
    with open(file, 'w', encoding='utf8') as f_out:
        for sentence in corpus_dict:
            sentence_lst = []
            for row in sentence:
                items = map(lambda x: row.get(x, '_'), column_names)
                sentence_lst += '\t'.join(items) + '\n'
            sentence_lst += '\n'
            f_out.write(''.join(sentence_lst))

def build_sequences(corpus_dict, key_x='form', key_y='ner', tolower=True):
    """
    Creates sequences from a list of dictionaries
    :param corpus_dict:
    :param key_x:
    :param key_y:
    :return:
    """
    X = []
    Y = []
    for sentence in corpus_dict:
        x = [word[key_x] for word in sentence]
        y = [word[key_y] for word in sentence]
        if tolower:
            x = list(map(str.lower, x))
        X += [x]
        Y += [y]
    return X, Y


def load(file):
    """
    Return the embeddings in the from of a dictionary
    :param file:
    :return:
    """
    file = file
    embeddings = {}
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:])
        embeddings[word] = vector
    glove.close()
    embeddings_dict = embeddings
    embedded_words = sorted(list(embeddings_dict.keys()))
    return embeddings_dict


embedding_file = '/Users/simonakesson/PycharmProjects/EDAN95Assignment1/EDANlab4/glove.6B.100d.txt'
embeddings_dict = load(embedding_file)

try:
    train_dict = pickle.load(open("train_dict.p", "rb"))
except FileNotFoundError:
    print('error')

try:
    dev_dict = pickle.load(open("dev_dict.p", "rb"))
except FileNotFoundError:
    print('error')

try:
    test_dict = pickle.load(open("test_dict.p", "rb"))
except FileNotFoundError:
    print('error')


class ContextDictorizer():
    """
    Extract contexts of words in a sequence
    Contexts are of w_size to the left and to the right
    Builds an X matrix in the form of a dictionary
    and possibly extracts the output, y, if not in the test step
    If the test_step is True, returns y = []
    """

    def __init__(self, input='form', output='ner', w_size=2, tolower=True):
        self.BOS_symbol = '__BOS__'
        self.EOS_symbol = '__EOS__'
        self.input = input
        self.output = output
        self.w_size = w_size
        self.tolower = tolower
        # This was not correct as the names were not sorted
        # self.feature_names = [input + '_' + str(i)
        #                     for i in range(-w_size, w_size + 1)]
        # To be sure the names are ordered
        zeros = math.ceil(math.log10(2 * W_SIZE + 1))
        self.feature_names = [input + '_' + str(i).zfill(zeros) for
                              i in range(2 * W_SIZE + 1)]

    def fit(self, sentences):
        """
        Build the padding rows
        :param sentences:
        :return:
        """
        self.column_names = sentences[0][0].keys()
        start = [self.BOS_symbol] * len(self.column_names)
        end = [self.EOS_symbol] * len(self.column_names)
        start_token = Token(dict(zip(self.column_names, start)))
        end_token = Token(dict(zip(self.column_names, end)))
        self.start_rows = [start_token] * self.w_size
        self.end_rows = [end_token] * self.w_size

    def transform(self, sentences, training_step=True):
        X_corpus = []
        y_corpus = []
        for sentence in sentences:
            X, y = self._transform_sentence(sentence, training_step)
            X_corpus += X
            if training_step:
                y_corpus += y
        return X_corpus, y_corpus

    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences)

    def _transform_sentence(self, sentence, training_step=True):
        # We extract y
        if training_step:
            y = [row[self.output] for row in sentence]
        else:
            y = None

        # We pad the sentence
        sentence = self.start_rows + sentence + self.end_rows

        # We extract the features
        X = list()
        for i in range(len(sentence) - 2 * self.w_size):
            # x is a row of X
            x = list()
            # The words in lower case
            for j in range(2 * self.w_size + 1):
                if self.tolower:
                    x.append(sentence[i + j][self.input].lower())
                else:
                    x.append(sentence[i + j][self.input])
            # We represent the feature vector as a dictionary
            X.append(dict(zip(self.feature_names, x)))
        return X, y

    def print_example(self, sentences, id=1):
        """
        :param corpus:
        :param id:
        :return:
        """
        # We print the features to check they match Table 8.1 in my book (second edition)
        # We use the training step extraction with the dynamic features
        Xs, ys = self._transform_sentence(sentences[id])
        print('X for sentence #', id, Xs)
        print('y for sentence #', id, ys)


def build_for_padd_dev(word_set, ner_set):
    X_words, Y_ner = build_sequences(dev_dict)
    rev_word_idx = dict(enumerate(word_set, start=2))
    rev_ner_idx = dict(enumerate(ner_set, start=2))
    word_idx = {v: k for k, v in rev_word_idx.items()}
    ner_idx = {v: k for k, v in rev_ner_idx.items()}
    X_words_idx = [list(map(lambda x: word_idx.get(x, 1), x)) for x in X_words]
    Y_ner_idx1 = [list(map(lambda x: ner_idx.get(x, 1), x)) for x in Y_ner]
    X_words_idx = pad_sequences(X_words_idx, maxlen=MAX_LEN)
    Y_ner_idx = pad_sequences(Y_ner_idx1, maxlen=MAX_LEN)
    print(X_words_idx[1])
    print(Y_ner_idx[1])
    return X_words_idx, Y_ner_idx, Y_ner_idx1


def build_for_padd_test(word_set, ner_set):
    X_words, Y_ner = build_sequences(test_dict)
    rev_word_idx = dict(enumerate(word_set, start=2))
    rev_ner_idx = dict(enumerate(ner_set, start=2))
    word_idx = {v: k for k, v in rev_word_idx.items()}
    ner_idx = {v: k for k, v in rev_ner_idx.items()}
    X_words_idx1 = [list(map(lambda x: word_idx.get(x, 1), x)) for x in X_words]
    Y_ner_idx1 = [list(map(lambda x: ner_idx.get(x, 1), x)) for x in Y_ner]
    X_words_idx = pad_sequences(X_words_idx1, maxlen=MAX_LEN)
    Y_ner_idx = pad_sequences(Y_ner_idx1, maxlen=MAX_LEN)
    print(X_words_idx[1])
    print(Y_ner_idx[1])
    return X_words_idx, Y_ner_idx, Y_ner_idx1

def build_for_padd_test1(word_set, ner_set):
    X_words, Y_ner = build_sequences(test_dict)
    rev_word_idx = dict(enumerate(word_set, start=2))
    rev_ner_idx = dict(enumerate(ner_set, start=2))
    word_idx = {v: k for k, v in rev_word_idx.items()}
    ner_idx = {v: k for k, v in rev_ner_idx.items()}
    X_words_idx1 = [list(map(lambda x: word_idx.get(x, 1), x)) for x in X_words]
    Y_ner_idx1 = [list(map(lambda x: ner_idx.get(x, 1), x)) for x in Y_ner]
   # X_words_idx = pad_sequences(X_words_idx1, maxlen=MAX_LEN)
   # Y_ner_idx = pad_sequences(Y_ner_idx1, maxlen=MAX_LEN)
    print(X_words_idx1[1])
    print(Y_ner_idx1[1])
    return X_words_idx1, Y_ner_idx1


def write_to_file(file, words, predicted, answer, rev_idx, rev_word):
    with open(file, 'w', encoding='utf8') as f_out:
        sentence_lst = []
        for i in range(2, len(words)):
            if words[i] > 1 and predicted[i] > 1 and answer[i] > 1:
                sentence_lst += rev_word[words[i]]
                sentence_lst += ' '
                sentence_lst += rev_idx[predicted[i]]
                sentence_lst += ' '
                sentence_lst += rev_idx[answer[i]]
                sentence_lst += '\n'
            else:
                continue

        f_out.write(''.join(sentence_lst))


def build_for_padd_training(selected_dict, words_gloVe):
    X_words, Y_ner = build_sequences(selected_dict)
    x_tot = X_words.copy()
    for word in words_gloVe:
        temp = []
        temp.append(word)
        x_tot.append(temp)
    word_set = sorted(list(set([item for sublist in x_tot for item in sublist])))
    ner_set = sorted(list(set([item for sublist in Y_ner for item in sublist])))
    rev_word_idx = dict(enumerate(word_set, start=2))
    rev_ner_idx = dict(enumerate(ner_set, start=2))
    word_idx = {v: k for k, v in rev_word_idx.items()}
    ner_idx = {v: k for k, v in rev_ner_idx.items()}
    X_words_idx = [list(map(lambda x: word_idx.get(x, 1), x)) for x in X_words]
    Y_ner_idx2 = [list(map(lambda x: ner_idx.get(x, 1), x)) for x in Y_ner]
    X_words_idx = pad_sequences(X_words_idx, maxlen=MAX_LEN)
    Y_ner_idx = pad_sequences(Y_ner_idx2, maxlen=MAX_LEN)
    print(X_words_idx[1])
    print(Y_ner_idx[1])
    return X_words_idx, Y_ner_idx, word_set, ner_set, Y_ner_idx2, rev_ner_idx, rev_word_idx, word_set, word_idx


column_names = ['form', 'ppos', 'pchunk', 'ner']

embeddings_words = embeddings_dict.keys()
print('Words in GloVe:', len(embeddings_dict.keys()))

X_train, y_train, w_set, n_set, y_train1, rev_ner, rev_word, word_set, word_index = build_for_padd_training(train_dict,
                                                                                                            embeddings_words)
X_val, y_val, y_val1 = build_for_padd_dev(w_set, n_set)
X_test1, y_test1, y_test2 = build_for_padd_test(w_set, n_set)

embedding_matrix = np.random.random((len(word_set) + 2,
                                     EMBEDDING_DIM))
for word in word_set:
    if word in embeddings_dict:
        # If the words are in the embeddings, we fill them with a value
        embedding_matrix[word_index[word]] = embeddings_dict[word]

print(len(word_set))
model = models.Sequential()
model.add(layers.Embedding(len(word_set) + 2, EMBEDDING_DIM,
                           input_length=150, mask_zero=True))
if embedding_matrix is not None:
    model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True
model.add(SimpleRNN(32, return_sequences=True))
model.add(Dropout(0.5))
model.add(layers.Dense(10, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)

y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
print(y_train.shape)
print(X_train.shape)
print(y_val.shape)

try:
    model = pickle.load(open("model10.p", "rb"))
except FileNotFoundError:
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
              callbacks=[checkpoint])
    pickle.dump(model, open("model10.p", "wb"))
y_test1 = to_categorical(y_test1)
test_loss, test_acc = model.evaluate(X_test1, y_test1)

X_test1, y_test1, y_test2 = build_for_padd_test(w_set, n_set)
pred = model.predict(X_test1)
pred1 = []
for predicted in pred:
    for subpred in predicted:
        pred1.append(np.argmax(subpred))
predicted_class_indices = np.argmax(pred, axis=1)
y_test2 = [j for i in y_test2 for j in i]
x_test1 = list(X_test1.flatten())
xtest = []
prd = []
for i in range(len(x_test1)):
    if x_test1[i] != 0:
        xtest.append(x_test1[i])
        prd.append(pred1[i])
write_to_file('outBase', xtest, prd, y_test2, rev_ner, rev_word)
print('Optimizer', OPTIMIZER, 'Epochs', EPOCHS, 'Batch size',
      BATCH_SIZE, 'Mini corpus', MINI_CORPUS)
print('Loss:', test_loss)
print('Accuracy:', test_acc)
print()
