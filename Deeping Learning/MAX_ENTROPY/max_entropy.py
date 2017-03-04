import tensorflow as tf
import data_helper
from tensorflow.contrib import learn
import numpy as np
import os
import time
import datetime

def compute_accurcy(x_data, y_data):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(input_y, 1))
    accurcy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return sess.run(accurcy, feed_dict={input_x: x_data, input_y: y_data})


learning_rate = 0.1
train_steps = 100
pos_file = "pos.txt"
neg_file = "neg.txt"
dev_sample_percentage = .1
display_steps = 1

print("loading data...")
x_text, y = data_helper.load_data_and_labels(pos_file, neg_file)

max_document_length = max([len(line.split(" ")) for line in x_text])
print("max_document_length = ", max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
#print(x)
#Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

W = tf.Variable(tf.random_normal([max_document_length, 2]))
# b = tf.Variable(tf.random_normal([2]))

input_x = tf.placeholder(tf.float32, [None, max_document_length])
input_y = tf.placeholder(tf.float32, [None, 2])

#model
pred = tf.nn.softmax(tf.matmul(input_x, W))

# cross_entropy = -tf.reduce_sum(input_y*tf.log(tf.clip_by_value(pred, 1e-10, 1.0)), reduction_indices=[1])
# loss = tf.reduce_mean(cross_entropy)
loss = tf.reduce_mean(-tf.reduce_sum(input_y*tf.log(tf.clip_by_value(pred,1e-10,1.0)), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    max_accurcy = .0

    #tensorboard   在终端用命令  $tensorboard --logdir=/tmp/tensorflowlogs  根据提示地址可显示
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", sess.graph)

    for step in range(train_steps + 1):
        sess.run(train, feed_dict={input_x: x_train, input_y: y_train})
        cur_accurcy = compute_accurcy(x_dev, y_dev)
        if cur_accurcy > max_accurcy:
            max_accurcy = cur_accurcy
        if step % display_steps == 0:
            print("step:","%04d" %(step),
                  "loss=",
                  "{:.8f}".format(sess.run(loss, feed_dict={input_x: x_train, input_y: y_train})),
                  "Accurcy:", cur_accurcy)
    print("Max_accurcy:", max_accurcy)
