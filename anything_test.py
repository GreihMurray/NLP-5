# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:45:03 2022

@author: omars
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Conv1D, MaxPool1D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, TimeDistributed, Attention
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import keras
from keras import models, layers
from keras import backend
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
from keras_self_attention import SeqSelfAttention
import collections
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential

def read_train_file(file_name):
    all_data = []
    descript = 'Reading ' + file_name

    f = open(file_name, 'r', encoding='utf-8')
    full_text = f.read()

    cur_sent = []

    for line in tqdm(full_text.split('\n'), desc=descript):
        if line == '<s>':
            cur_sent = []
            continue
        if line in '()':
            continue
        if line == '</s>':
            all_data.append(cur_sent)
            continue
        else:
            cur_sent.append(line.lower())

    return all_data

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def build_model(input_shape, output_sequence_length, source_vocab_size, target_vocab_size):

    model = Sequential()
    model.add(Embedding(input_dim=source_vocab_size,output_dim=256,input_length=input_shape[1]))
    model.add(Bidirectional(LSTM(256, activation="tanh",return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='softmax'))
    # model.add((LSTM(512, activation = 'relu')))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add((Dense(target_vocab_size,activation='softmax')))
    learning_rate = 0.005

    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    source = read_train_file('train-source.txt')
    target = read_train_file('train-target.txt')

    source_counter = collections.Counter([word for sentence in source for word in sentence])
    target_counter = collections.Counter([word for sentence in target for word in sentence])

    print('{} Source words.'.format(len([word for sentence in source for word in sentence])))
    print('{} unique source words.'.format(len(source_counter)))
    print('10 Most common words in the source dataset:')
    print('"' + '" "'.join(list(zip(*source_counter.most_common(10)))[0]) + '"')
    print()
    print('{} Target words.'.format(len([word for sentence in target for word in sentence])))
    print('{} unique target words.'.format(len(target_counter)))
    print('10 Most common words in the target dataset:')
    print('"' + '" "'.join(list(zip(*target_counter.most_common(10)))[0]) + '"')

    pre_source, pre_target, source_token, target_token =\
    preprocess(source, target)

    max_source_length = pre_source.shape[1]
    max_target_length = pre_target.shape[1]
    source_vocab_size = len(source_token.word_index)
    target_vocab_size = len(target_token.word_index)

    print('Data Preprocessed')
    print("Max source sentence length:", max_source_length)
    print("Max target sentence length:", max_target_length)
    print("Source vocabulary size:", source_vocab_size)
    print("Target vocabulary size:", target_vocab_size)

    # Train Simple RNN

    tmp_x = pad(pre_source, max_target_length)
    tmp_x = tmp_x.reshape((-1, pre_target.shape[-2], 1))

    # Train the neural network
    simple_rnn_model = build_model(
        tmp_x.shape,
        max_target_length,
        source_vocab_size,
        target_vocab_size)
    simple_rnn_model.fit(tmp_x, pre_target, batch_size=32, epochs=10, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], target_token))

    # simple_rnn_model.save('attention')
    new_model = tf.keras.models.load_model('attention')
    print(logits_to_text(new_model.predict(tmp_x[:1])[0], source_token))
    predictions = new_model.predict(tmp_x[:1])[0]
    one = tmp_x[0]
    new_model.evaluate(tmp_x,pre_target)
    index_to_words = {id: word for word, id in target_token.word_index.items()}
    index_to_words[3]
