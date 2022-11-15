# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:24:48 2022

@author: omars
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

def read_train_file(file_name):
    all_data = []
    descript = 'Reading ' + file_name

    f = open(file_name, 'r', encoding='utf-8')
    full_text = f.read()

    cur_sent = []

    for line in tqdm(full_text.split('\n'), desc=descript):
        if line == '<s>':
            cur_sent = ''
            continue
        if line in '()':
            continue
        if line == '</s>':
            all_data.append(cur_sent)
            continue
        else:
            cur_sent += line.lower()
            cur_sent += ' '

    return all_data


if __name__ == "__main__":
    input_texts = read_train_file('train-source.txt')
    target_texts = read_train_file('train-target.txt')
    input_characters = set()
    target_characters = set()

    for sentence in input_texts:
        for char in sentence:
            if char not in input_characters:
                input_characters.add(char)
    for sentence in target_texts:
        for char in sentence:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0