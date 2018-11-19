# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import importlib
import notMNIST_methods as nmm
importlib.reload(nmm)
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
image_size = 28
num_labels = 10
num_channels = 1  # grayscale


train_dataset, train_labels = nmm.reformat_cube(train_dataset, train_labels)
valid_dataset, valid_labels = nmm.reformat_cube(valid_dataset, valid_labels)
test_dataset, test_labels = nmm.reformat_cube(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()
dimension_list = [[patch_size, patch_size, num_channels, depth], [patch_size, patch_size,
                                                                  depth, depth], [image_size//4*image_size//4*depth, num_hidden], [num_hidden, num_labels]]

dimension_list[0][-1]
with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    def generate_variables(dimension_list):
        weights = []
        biases = []
        for i in range(len(dimension_list)):
            weights.append(tf.Variable(tf.truncated_normal(
                dimension_list[i], stddev=0.1)))
            biases.append(tf.Variable(tf.zeros([dimension_list[i][-1]])))
        return weights, biases

    weights, biases = generate_variables(dimension_list)

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, weights[0], [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + biases[0])
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, weights[1], [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + biases[1])
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, weights[2]) + biases[2])
        return tf.matmul(hidden, weights[3]) + biases[3]

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


#### SESSION ###########

num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % nmm.accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % nmm.accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % nmm.accuracy(test_prediction.eval(), test_labels))


######## RESHAPE #######
train_dataset = np.pad(train_dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
valid_dataset = np.pad(valid_dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
test_dataset = np.pad(test_dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

image_size = 32
num_labels = 10
num_channels = 1  # grayscale
batch_size = 128
patch_size = 5
depth_1 = 6
depth_2 = 16
num_hidden = 120


graph = tf.Graph()
dimension_list = [[patch_size, patch_size, num_channels, depth_1], [patch_size, patch_size, depth_1, depth_2], [25*depth_2, num_hidden], [num_hidden, num_labels]]
print(dimension_list)
with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)

    # Variables.
    def generate_variables(dimension_list):
        weights = []
        biases = []
        for i in range(len(dimension_list)):
            weights.append(tf.Variable(tf.truncated_normal(
                dimension_list[i], mean=0, stddev=0.1)))
            biases.append(tf.Variable(tf.zeros([dimension_list[i][-1]])))
        return weights, biases

    weights, biases = generate_variables(dimension_list)

    # Model.
    def model(data, prob=1.0):
        # 32x32
        #I can probably use same padding and not pad the images
        c1 = tf.nn.conv2d(data, weights[0], [1, 1, 1, 1], padding='VALID')
        c1 = tf.nn.relu(c1 + biases[0])
        c1 = tf.nn.dropout(c1, prob)
        # 28x28
        s2 = tf.nn.max_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # 14x14
        c3 = tf.nn.conv2d(s2, weights[1], [1, 1, 1, 1], padding='VALID')
        c3 = tf.nn.relu(c3 + biases[1])
        c3 = tf.nn.dropout(c3, prob)
        # 10x10
        s4 = tf.nn.max_pool(c3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # 5x5
        shape = s4.get_shape().as_list()
        c5 = tf.reshape(s4, [shape[0], shape[1] * shape[2] * shape[3]])
        c5 = tf.nn.relu(tf.matmul(c5, weights[2]) + biases[2])
        c5 = tf.nn.dropout(c5, prob)
        return tf.matmul(c5, weights[3]) + biases[3]

    # Training computation.
    logits = model(tf_train_dataset, 0.8)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    # Optimizer.
    #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    #learning_rate = tf.train.exponential_decay(0.5, global_step, 5000, 0.8, staircase=True)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    rate=0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
    #training_operation = optimizer.minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


#### SESSION ###########

num_steps = 20001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % nmm.accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % nmm.accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % nmm.accuracy(test_prediction.eval(), test_labels))
