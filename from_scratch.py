import json
import utility
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# trans_probs: no 1-to-2 or 2-to-1 (0.678 BLEU)

# trans_probs_adjust: 1-to-2 no 2-to-1 (0.6797 BLEU)

# trans_probs_both: 1-to-2 and 2-to-1 (0.6794 BLEU)


def accuracy(preds, test):
    total_words = 0
    total_right = 0

    for i in range(0, len(preds)):
        total_words += len(preds[i])
        total_right += len(set(preds[i]) & set(test[i]))

    return total_right / total_words


def main():
    source = utility.read_train_file('train-source.txt')
    target = utility.read_train_file('train-target.txt')

    utility.save_sents(source, target)

    source = utility.clean_dash(source)

    s_train = source[:int(len(source) * 0.8)]
    s_test = source[int(len(source) * 0.8):]

    t_train = target[:int(len(target) * 0.8)]
    t_test = target[int(len(target) * 0.8):]

    counts = utility.naive_counts(s_train, t_train, 3)
    probs = utility.naive_probs(counts)

    utility.target_probs(t_train)

    def_source, def_targ = utility.read_defs()

    preds = utility.predict(s_test, probs, def_source, def_targ)

    bleu = utility.bleu_score(preds, t_test)
    print(accuracy(preds, t_test) * 100)


def evaluate():
    source = utility.read_train_file('train-source.txt')
    target = utility.read_train_file('train-target.txt')

    def_source, def_targ = utility.read_defs()

    # source = utility.clean_dash(source)

    source = source[int(len(source) * 0.8):]
    target = target[int(len(target) * 0.8):]

    probs = {}

    with open('trans_probs_both.json', 'r') as infile:
        probs = json.load(infile)

    preds = utility.predict(source, probs, def_source, def_targ)

    new_targ = []

    for row in target:
        new_targ.append([row])

    bleu = utility.bleu_score(preds, target, MAX_N=4)
    print('Custom BLEU: ', bleu)
    print(corpus_bleu(new_targ, preds))
    print(accuracy(preds, target) * 100)


if __name__ == '__main__':
    evaluate()
