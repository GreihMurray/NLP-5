import json
import utility
from nltk.translate.bleu_score import corpus_bleu

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

    source = utility.clean_dash(source)

    s_train = source[:int(len(source) * 0.8)]
    s_test = source[int(len(source) * 0.8):]

    t_train = target[:int(len(target) * 0.8)]
    t_test = target[int(len(target) * 0.8):]

    counts = utility.naive_counts(s_train, t_train, 3)
    probs = utility.naive_probs(counts)

    utility.target_probs(t_train)

    preds = utility.predict(s_test, probs)

    bleu = utility.bleu_score(preds, t_test)
    print(accuracy(preds, t_test) * 100)


def evaluate():
    source = utility.read_train_file('train-source.txt')
    target = utility.read_train_file('train-target.txt')

    # source = utility.clean_dash(source)

    source = source[int(len(source) * 0.8):]
    target = target[int(len(target) * 0.8):]

    probs = {}

    with open('trans_probs.json', 'r') as infile:
        probs = json.load(infile)

    preds = utility.predict(source, probs)

    bleu = utility.bleu_score(preds, target, MAX_N=4)
    print('Custom BLEU: ', bleu)
    print(accuracy(preds, target) * 100)


if __name__ == '__main__':
    evaluate()
