import csv
from tqdm import tqdm
import math

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


def naive_counts(source, target):
    counts = {}

    for i in tqdm(range(0, len(source)), desc='Counting occurences'):
        if len(source[i]) != len(target[i]):
            continue
        for j in range(0, len(source[i])):
            source_word = source[i][j]

            target_word = target[i][j]

            if source_word in counts.keys():
                if target_word in counts[source_word].keys():
                    counts[source_word][target_word] += 1
                else:
                    counts[source_word][target_word] = 1
            else:
                counts[source_word] = {target_word: 1}

    return counts


def naive_probs(counts):
    probs = {}

    for key, data_dict in counts.items():
        probs[key] = {}
        dict_sum = sum(list(data_dict.values()))

        for new_key, value in data_dict.items():
            probs[key][new_key] = (counts[key][new_key] / dict_sum)

    return probs


def predict(s_test, probs):
    all_preds = []

    for sentence in tqdm(s_test, desc='Predicting'):
        cur_sent = []
        for word in sentence:
            try:
                cur_sent.append(max(probs[word], key=probs[word].get))
            except KeyError:
                continue

        all_preds.append(cur_sent)

    return all_preds


def bleu_score(source, target, MAX_N=4):
    score = 0

    for i in tqdm(range(0, len(source)), desc='Calculating Bleu score'):
        precision = 0

        for N in range(1, MAX_N):
            source_grams = [' '.join(source[i][j:j + N]) for j in range(len(source[i]) - N + 1)]
            target_grams = [' '.join(source[i][j:j + N]) for j in range(len(target[i]) - N + 1)]

            if len(source_grams) == 0:
                continue

            if N == 1:
                precision = (len(set(source_grams) & set(target_grams)) / len(source_grams)) ** (1/MAX_N)
                continue
            precision *= (len(set(source_grams) & set(target_grams)) / len(source_grams)) ** (1/MAX_N)

        brevity = 0

        if len(source[i]) == len(target[i]):
            brevity = 1
        else:
            try:
                brevity = math.exp((1-(len(target[i])/len(source[i]))))
            except ZeroDivisionError:
                brevity = math.exp((1 - (len(target[i]) / 1)))

        bleu = precision * brevity

        score += bleu

    return score
