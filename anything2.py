# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:24:48 2022

@author: omars
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM,Flatten
from keras.layers import Embedding
from keras.optimizers import Adam,RMSprop
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import OneHotEncoder

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

def tokenize(chars):
    tokened = {}
    for i, char in enumerate(chars):
        tokened[char] = i+1
    tokened[''] = 0
    return tokened

def tokenize_sentences(sentences, indexes):
    tokenized_all = []
    for sentence in sentences:
        tokenized = []
        for char in sentence:
            tokenized.append(indexes[char])
        tokenized_all.append(tokenized)
    return tokenized_all

def pad(sentences, length):
    return pad_sequences(sentences, maxlen = length, padding = 'post')

def preprocess(text, chars, max_len):
    token_index = tokenize(chars)

    tokenized_text = tokenize_sentences(text, token_index)

    padded_text = pad(tokenized_text, max_len)

    padded_text = padded_text.reshape((padded_text.shape[0],padded_text.shape[1],1))

    return padded_text,token_index

def decode(logits, tokenizer):

    return ' '.join([list(tokenizer.values()).index(prediction) for prediction in np.argmax(logits, 1)])

def build_model(input_shape, output_sequence_length, source_vocab_size, target_vocab_size):

    # model = Sequential()
    # model.add(Embedding(input_dim=source_vocab_size,output_dim=256,input_length=input_shape[1]))
    # model.add(Bidirectional(LSTM(256, activation="tanh",return_sequences=True)))
    # model.add(Flatten())
    # model.add(SeqSelfAttention(attention_activation='softmax'))
    # # model.add((LSTM(512, activation = 'tanh')))
    # model.add(Dense(512, activation = 'relu'))
    # model.add(Dense(128, activation = 'relu'))
    # model.add(Dense(64, activation = 'relu'))
    # model.add((Dense(target_vocab_size,activation='softmax')))
    # learning_rate = 0.009

    # model.compile(loss = 'sparse_categorical_crossentropy',
    #               optimizer = RMSprop(learning_rate),
    #               metrics = ['accuracy'])

    model = Sequential()
    model.add(Embedding(input_dim=source_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(target_vocab_size,activation='softmax')))
    learning_rate = 0.005

    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    source_texts = read_train_file('train-source.txt')
    target_texts = read_train_file('train-target.txt')
    source_characters = set()
    target_characters = set()

    for sentence in source_texts:
        for char in sentence:
            if char not in source_characters:
                source_characters.add(char)
    for sentence in target_texts:
        for char in sentence:
            if char not in target_characters:
                target_characters.add(char)

    source_characters = sorted(list(source_characters))
    target_characters = sorted(list(target_characters))
    num_source_tokens = len(source_characters)
    num_target_tokens = len(target_characters)
    max_source_seq_length = max([len(txt) for txt in source_texts])
    max_target_seq_length = max([len(txt) for txt in target_texts])
    max_sentence_length = max(max_source_seq_length, max_target_seq_length)

    print("Number of samples:", len(source_texts))
    print("Number of unique source tokens:", num_source_tokens)
    print("Number of unique output tokens:", num_target_tokens)
    print("Max sequence length for sources:", max_source_seq_length)
    print("Max sequence length for outputs:", max_target_seq_length)

    processed_source, source_tokens = preprocess(source_texts, source_characters, max_sentence_length)
    processed_target, target_tokens = preprocess(target_texts, target_characters, max_sentence_length)

    model = build_model(processed_source.shape, max_sentence_length, num_source_tokens, num_target_tokens)

    model.fit(processed_source, processed_target, batch_size=32, epochs=1, validation_split=0.2)
    model.save('test')

    predicted_sentence = model.predict(processed_source[:10])
    decode(predicted_sentence,target_tokens)





