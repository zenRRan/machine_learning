import numpy as np
import re
import itertools
from collections import Counter

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
        open(positive_data_file, "r").readlines()
    )
    positive_examples = [s.strip() for s in positive_examples]
    negitive_examples = list(
        open(negitive_data_file, "r").readlines()
    )
    negitive_examples = [s.strip() for s in negitive_examples]

    x_data = positive_examples + negitive_examples
    x_data = [clean_str(sent) for sent in x_data]
    # print(x_data)
    positive_labels = [[0, 1] for _ in positive_examples]
    negvitive_labels = [[1, 0] for _ in negitive_examples]
    # print(positive_labels)
    y = np.concatenate([positive_labels, negvitive_labels], 0)
    # print(x_data)
    # print(y)
    return [x_data, y]