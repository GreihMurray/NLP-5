import utility


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

    preds = utility.predict(s_test, probs)

    bleu = utility.bleu_score(preds, t_test)
    print(bleu)

    print(accuracy(preds, t_test) * 100)


if __name__ == '__main__':
    main()