# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:36:12 2022

@author: omars
"""
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

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    # TODO: Build the layers
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences = True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])

    return model

def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    # TODO: Implement
    learning_rate = 1e-3
    rnn = GRU(64, return_sequences=True, activation="tanh")

    embedding = Embedding(french_vocab_size, 64, input_length=input_shape[1])
    logits = TimeDistributed(Dense(french_vocab_size, activation="softmax"))

    model = Sequential()
    #em can only be used in first layer --> Keras Documentation
    model.add(embedding)
    model.add(rnn)
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    learning_rate = 1e-3
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences = True, dropout = 0.1),
                           input_shape = input_shape[1:]))
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])
    return model

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # OPTIONAL: Implement
    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape = input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences = True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))

    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])
    return model

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    learning_rate = 0.005

    model.compile(loss = sparse_categorical_crossentropy,
                 optimizer = Adam(learning_rate),
                 metrics = ['accuracy'])

    return model

def attention_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
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
    simple_rnn_model = simple_model(
        tmp_x.shape,
        max_target_length,
        source_vocab_size,
        target_vocab_size)
    simple_rnn_model.fit(tmp_x, pre_target, batch_size=32, epochs=10, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], target_token))

    simple_rnn_model.save('simple_rnn')

    # Train bi-directional

    tmp_x = pad(pre_source, pre_target.shape[1])
    tmp_x = tmp_x.reshape((-1, pre_target.shape[-2], 1))

    bidi_model = bd_model(
        tmp_x.shape,
        pre_target.shape[1],
        len(source_token.word_index)+1,
        len(target_token.word_index)+1)


    bidi_model.fit(tmp_x, pre_target, batch_size=32, epochs=20, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(bidi_model.predict(tmp_x[:1])[0], target_token))

    bidi_model.save('bidirection')

    # Train embedding model

    tmp_x = pad(pre_source, max_target_length)
    tmp_x = tmp_x.reshape((-1, pre_target.shape[-2]))

    # TODO: Train the neural network

    embeded_model = embed_model(
        tmp_x.shape,
        max_target_length,
        source_vocab_size,
        target_vocab_size)

    embeded_model.fit(tmp_x, pre_target, batch_size=32, epochs=10, validation_split=0.2)


    # TODO: Print prediction(s)
    print(logits_to_text(embeded_model.predict(tmp_x[:1])[0], target_token))

    embeded_model.save('embedding')

    # Train encode/decode model

    tmp_x = pad(pre_source)
    tmp_x = tmp_x.reshape((-1, pre_target.shape[-2]))

    encodeco_model = encdec_model(
        tmp_x.shape,
        pre_source.shape[1],
        len(source_token.word_index)+1,
        len(target_token.word_index)+1)

    encodeco_model.fit(tmp_x, pre_target, batch_size=32, epochs=20, validation_split=0.2)

    print(logits_to_text(encodeco_model.predict(tmp_x[:1])[0], target_token))

    encodeco_model.save('encode')

    # Train Embedding Bi-directional model

    tmp_X = pad(pre_source)
    model = model_final(tmp_X.shape,
                        pre_target.shape[1],
                        len(source_token.word_index)+1,
                        len(target_token.word_index)+1)

    model.fit(tmp_X, pre_target, batch_size = 32, epochs = 17, validation_split = 0.2)

    model.save('bidi-encode')