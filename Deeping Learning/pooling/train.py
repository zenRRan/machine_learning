import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from pooling import data_halper

def embeded_sum_pooling(W, data):
    buffer = []
    for line in data:
        buffer.append(tf.reduce_sum(tf.nn.embedding_lookup(W, input), 0))
    return buffer

tf.flags.DEFINE_string("train_path","/Users/zhenranran/Desktop/train.txt","train path")
tf.flags.DEFINE_integer("class_number",3,"class number")
tf.flags.DEFINE_integer("embeding_size",128,"embeding size")
tf.flags.DEFINE_float("learning_rate",0.08,"learning rate")
tf.flags.DEFINE_integer("train_steps",1000,"train steps")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nPatamters:\n")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(),value))
print("")

'''load data'''
print("loading data...")
x, y, vocabs = data_halper.read_file(FLAGS.train_path)
print("done")

'''one-hot'''
print("\none-hot...")
max_document_lenth = max([len(text.split()) for text in x])
print("max_document_lenth=",max_document_lenth)
vocab_process = learn.preprocessing.VocabularyProcessor(max_document_lenth)
x = np.array(list(vocab_process.fit_transform(x)))
# print(x)
print("done")

'''tensorflow'''

input_x = tf.placeholder(tf.int32, [None, max_document_lenth])
input_y = tf.placeholder(tf.float32, [None, FLAGS.class_number])

# embedings_W = tf.Variable(tf.random_normal([1000, FLAGS.embeding_size]))
embedings_W = tf.Variable(
    tf.random_normal([1500, FLAGS.embeding_size], -0.1, 0.1,dtype=tf.float32)
)

#sum embeded data
input = tf.reduce_sum(
    tf.reduce_sum(
        tf.nn.embedding_lookup(embedings_W, input_x),
        0),
    0)
input_ = tf.reshape(input, [1, 128])

# W = tf.Variable(tf.random_normal([FLAGS.embeding_size, FLAGS.class_number]))
W = tf.Variable(
    tf.random_normal([FLAGS.embeding_size, FLAGS.class_number], -0.1, 0.1,dtype=tf.float32)
)
mul = tf.matmul(input_, W)
pred = tf.nn.softmax(mul)
# loss = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(tf.clip_by_value(pred,1e-10,1.0)), reduction_indices=[1]))
loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=input_y, logits=mul
) )

train = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    print("\ninit...")
    sess.run(init)
    print("done\n")
    print("Training...")

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/tensorflowlogs", sess.graph)

    for step in range(FLAGS.train_steps):
        # print("Step:", step)
        # print("------------------------------------------------------------------------------------------------------------")
        error = 0
        i = 0
        for line, y_ in zip(x,y):
            # if i % 100 == 0:
            #     print(i)
            # i += 1
            y_ = np.matrix(y_)
            line = np.matrix(line)
            # print(sess.run([train, pred, loss, embedings_W, W], feed_dict={input_x: line, input_y: y_}))
            sess.run(train, feed_dict={input_x: line, input_y: y_})
        #print("evalution...\n")
        i = 0
        for line, y_ in zip(x,y):
            # if i % 100 == 0:
            #     print(i)
            # i += 1
            line = np.matrix(line)
            y_ = np.matrix(y_)
            correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
            _, C = sess.run([pred, accuracy], feed_dict={input_x: line, input_y: y_})
            error += C
        print(step," accuracy: ", error / len(y))
    print("Train is over!")




    # mat_x = np.matrix(x)
    # mat_y = np.matrix(y)
    # for step in range(FLAGS.train_steps):
    #     print("Step:", step)
    #     print("------------------------------------------------------------------------------------------------------------")
    #     print(sess.run([train, loss], feed_dict={input_x: mat_x, input_y: mat_y}))
    #     correct_predictions = tf.equal(pred, tf.argmax(input_y, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    #     print("accuracy: ", accuracy)

