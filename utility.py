from tqdm import tqdm
import math
import json

PUNCT = ',.'


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


def clean_dash(data):
    clean_data = []

    for sentence in tqdm(data, desc='Cleaning'):
        skip = False
        cur_sent = []
        for i in range(0, len(sentence)-1):
            if skip:
                skip = False
                continue
            cur_word = sentence[i]
            if cur_word == '-':
                if i == len(sentence):
                    continue
                if len(cur_sent) < 1:
                    continue
                else:
                    cur_word = ''.join([sentence[i-1], sentence[i+1]])
                    del cur_sent[-1]
                    skip = True
            cur_sent.append(cur_word)
        clean_data.append(cur_sent)

    return clean_data


def target_probs(target):
    counts = {}

    for sentence in tqdm(target, desc='Target counts'):
        for i in range(0, len(sentence)):
            if sentence[i] in counts.keys():
                counts[sentence[i]] += 1
            else:
                counts[sentence[i]] = 1

    probs = {}

    for word in counts.keys():
        probs[word] = counts[word] / sum(list(counts.values()))

    with open('target_probs.json', 'w') as outfile:
        json.dump(probs, outfile)


def naive_counts(source, target, max_edit_check):
    counts = {}

    for i in tqdm(range(0, len(source)), desc='Counting occurrences'):
        offset = 0
        check_lim = 0
        skip = False
        for j in range(0, len(source[i])):
            source_word = source[i][j]

            if skip:
                skip = False
                continue

            if j > 1:
                if source[i][j-1] in PUNCT:
                    check_lim -= 1

            if j == len(source[i])-1:
                check_lim -= 1

            if offset+1 == max_edit_check and j > len(target[i]):
                continue

            dists = []
            for t in range(j+check_lim, min(j+max_edit_check, len(target[i]))):
                dist = levenshtein(source_word, target[i][t-offset])
                if source_word in target[i][t-offset]:
                    dist = min(dist, abs(len(source_word) - len(target[i][t-offset])))
                dists.append(dist)

            if j+1 != len(target[i]):
                grams = [''.join(target[i][j:j + 1])]
            gram_dists = []

            for gram in grams:
                gram_dists.append(levenshtein(source_word, gram))

            if not dists:
                dists.append(levenshtein(source_word, target[i][-1]))

            if min(dists) >= len(source_word):
                offset += 1
                continue

            # print(j, offset, len(target[i]), dists)
            # print((j+dists.index(min(dists))) - (offset - check_lim), source_word)
            # print(source[i])
            # print(target[i])
            if min(gram_dists) < min(dists):
                skip = True
                target_word = grams[grams.index(min(grams))]
            else:
                target_word = target[i][(j+dists.index(min(dists))) - (offset - check_lim)]
            # print(target_word)
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

    with open('trans_probs_adjust.json', 'w') as outfile:
        json.dump(probs, outfile)

    return probs


def predict(s_test, probs):
    all_preds = []

    trans_weight = 1
    targ_weight = 0

    with open('target_probs.json') as infile:
        targ_probs = json.load(infile)

    for sentence in tqdm(s_test, desc='Predicting'):
        cur_sent = []
        for word in sentence:
            try:
                max_prob = 0
                max_key = ''
                for key in probs[word]:
                    if key in targ_probs.keys():
                        prob = (probs[word][key] * trans_weight) + (targ_probs[key] * targ_weight)
                    else:
                        prob = (probs[word][key] * trans_weight)

                    if prob > max_prob:
                        max_prob = prob
                        max_key = key

                cur_sent.append(max_key)
                # cur_sent.append(max(probs[word], key=probs[word].get))
            except KeyError:
                continue

        all_preds.append(cur_sent)

    return all_preds


def levenshtein(word1, word2):
    if len(word1) > len(word2):
        word1, word2 = word2, word1

    dists = range(0, len(word1)+1)

    for index, char in enumerate(word2):
        dists_ = [index+1]

        for ind2, char2 in enumerate(word1):
            if char == char2:
                dists_.append(dists[ind2])
            else:
                dists_.append(1 + min(dists[ind2], dists[ind2+1], dists_[-1]))
        dists = dists_

    return dists[-1]


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

        if len(source[i]) > len(target[i]):
            brevity = 1
        else:
            try:
                brevity = math.exp((1-(len(target[i])/len(source[i]))))
            except ZeroDivisionError:
                brevity = math.exp((1 - (len(target[i]) / 1)))

        bleu = precision * brevity

        score += bleu

    score = (score / len(source))

    return score
