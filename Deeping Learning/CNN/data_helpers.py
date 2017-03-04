import numpy as np
import re
import itertools
from collections import Counter

chineseFile_train_dev = "/Users/zhenranran/PycharmProjects/classification/panda_data/train_dev.txt"
chineseFile_labels = "/Users/zhenranran/PycharmProjects/classification/panda_data/train_dev_labels.txt"
englishFile_Positive = "/Users/zhenranran/PycharmProjects/classification/rt-polaritydata/rt-polarity.pos"
englishFile_Negitive = "/Users/zhenranran/PycharmProjects/classification/rt-polaritydata/rt-polarity.neg"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file, negitive_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    positive_examples = list(
        open(englishFile_Positive, "r").readlines()
    )
    positive_examples = [s.strip() for s in positive_examples]
    negitive_examples = list(
        open(englishFile_Negitive, "r").readlines()
    )
    negitive_examples = [s.strip() for s in negitive_examples]

    x_data = positive_examples + negitive_examples
    x_data = [clean_str(sent) for sent in x_data]

    positive_labels = [[0, 1] for _ in positive_examples]
    negvitive_labels = [[1, 0] for _ in negitive_examples]
    print(positive_labels)
    y = np.concatenate([positive_labels, negvitive_labels], 0)
    print(x_data)
    print(y)
    return [x_data, y]
def chn_load_data_and_labels(train_dev_file, labels_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x_data = list(
        open(train_dev_file, "r").readlines()
    )
    x_data = [s.strip() for s in x_data]

    label = list(open(labels_file, "r").readlines())
    labels = []
    for l in label:
        if l == "p\n":
            labels.append([0, 0, 1])
        elif l == "m\n":
            labels.append([0, 1, 0])
        else:
            labels.append([1, 0, 0])
    # print(x_data)
    labels = np.concatenate([labels])
    # print(labels)
    return [x_data, labels]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
# load_data_and_labels(englishFile_Positive, englishFile_Negitive)
chn_load_data_and_labels(chineseFile_train_dev, chineseFile_labels)