import tensorflow as tf
import tensorflow.contrib.slim as slim

from tutorial_utils import *

single_motif_detection_simulation_parameters = {
    "motif_name": "TAL1_known4",
    "seq_length": 1000,
    "num_pos": 10000,
    "num_neg": 10000,
    "GC_fraction": 0.4}

single_motif_detection_simulation_data = get_simulation_data(
            "simulate_single_motif_detection", single_motif_detection_simulation_parameters)

data = single_motif_detection_simulation_data
X_train = data.X_train.transpose((0, 1, 3, 2))
X_valid = data.X_valid.transpose((0, 1, 3, 2))
X_test = data.X_test.transpose((0, 1, 3, 2))

y_train = data.y_train
y_valid = data.y_valid
y_test = data.y_test

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("X_test shape:", X_test.shape)


# Create the model

batch_size = 256
learning_rate = 0.001

x = tf.placeholder(tf.float32, [batch_size, 1, 1000, 4], name='input')
y = tf.placeholder(tf.float32, [batch_size, 1], name='output')

net = slim.conv2d(x, 10, [1, 15], scope='conv1_1')
net = slim.max_pool2d(net, [1, 5], stride=5, scope='pool1')

net = slim.conv2d(net, 10, [1, 10], scope='conv2_3')
net = slim.max_pool2d(net, [1, 10], stride=10, scope='pool2')

net = tf.reshape(net, [batch_size, 20*10])
net = slim.fully_connected(net, 1, activation_fn=None, scope='fc4')

loss = slim.losses.sigmoid_cross_entropy(net, y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = slim.learning.create_train_op(loss, optimizer)

logdir = './log'  # Where checkpoints are stored.

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(10000):
        samples = random.sample(range(X_train.shape[0]), batch_size)
        X_batch = X_train[samples]
        y_batch = y_train[samples]

        l, _ = sess.run([loss, train_op], feed_dict={x: X_batch, y: y_batch})
        print("Step %03d: loss %f" % (i, l))

