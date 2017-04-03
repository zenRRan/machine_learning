import numpy as np
import math
def read_file(file_path):
    data = open(file_path).readlines()
    # print(data)
    row = []
    labels = []
    s = {}
    for line in data:
        line = line.strip().split()
        s = set(line).union(s)
        if line[1] == "IphoneSE":
            row.append(" ".join(line[2:len(line) - 1]))
            labels.append(line[-1])
    label = []
    for line in labels:
       if line == "FAVOR":
           label.append([1, 0, 0])
       elif line == "AGAINST":
           label.append([0, 1, 0])
       else:
           label.append([0, 0, 1])
       # row = np.matrix(row)
       # label = np.matrix(label)
    # print(s)
    return row, label, len(s)

# _, _, d = read_file("/Users/zhenranran/Desktop/train.txt")
# print(d)
