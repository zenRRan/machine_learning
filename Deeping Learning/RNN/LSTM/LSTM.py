#tensorflow version 0.12

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

#downloading need network  or return error
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

learning_rate = 0.01
max_samples = 4000000
batchs_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hiddens = 256
n_classes = 10

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = tf.Variable(tf.random_normal([2*n_hiddens, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hiddens, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hiddens, forget_bias=1.0)

    outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batchs_size < max_samples:
        batchs_x, batchs_y = mnist.train.next_batch(batchs_size)
        batchs_x = batchs_x.reshape((batchs_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batchs_x, y:batchs_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x:batchs_x, y: batchs_y})
            loss = sess.run(cost, feed_dict={x:batchs_x, y: batchs_y})
            print("Iter " + str(step*batchs_size) + "，Minibatch Loss " + "{:.6f}".format(loss),
                  ", Training Accurary= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

