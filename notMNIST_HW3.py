# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

import notMNIST_methods as nmm

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

train_dataset, train_labels = nmm.reformat(train_dataset, train_labels)
valid_dataset, valid_labels = nmm.reformat(valid_dataset, valid_labels)
test_dataset, test_labels = nmm.reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


batch_size = 128
hidden_nodes = 1024
num_steps = 3001


def SGD_logits(train, weights, biases):
    logits = tf.matmul(train, weights[0]) + biases[0]
    return logits


def NN_logits(train, weights, biases):

    for i in range(len(weights)):
        logits = tf.matmul(train, weights[i]) + biases[i]
        train = tf.nn.relu(logits)
    return logits


dimension_list = [image_size*image_size, hidden_nodes, num_labels]
logit_maker = NN_logits
##################
# MODIFIED GRAPH
####################
graph = tf.Graph()
with graph.as_default():
    def generate_variables(dimension_list):
        weights = []
        biases = []
        for i in range(1, len(dimension_list)):
            weights.append(tf.Variable(tf.truncated_normal(
                [dimension_list[i-1], dimension_list[i]])))
            biases.append(tf.Variable(tf.zeros([dimension_list[i]])))
        return weights, biases

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta = tf.placeholder(tf.float32)

    # Variables.
    weights, biases = generate_variables(dimension_list)
    # Training computation.
    logits = logit_maker(tf_train_dataset,  weights, biases)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    for weight in weights:
        loss = loss + beta*tf.nn.l2_loss(weight)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_logits = logit_maker(tf_valid_dataset, weights, biases)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_logits = logit_maker(tf_test_dataset, weights, biases)
    test_prediction = tf.nn.softmax(test_logits)


######################
# Session
######################


beta_vals = [10**i for i in np.linspace(-3, -1, 21)]
test_acc = []
# beta_vals=[0]
for beta_val in beta_vals:
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels, beta: beta_val}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % nmm.accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % nmm.accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % nmm.accuracy(test_prediction.eval(), test_labels))
        test_acc.append(nmm.accuracy(test_prediction.eval(), test_labels))
plt.semilogx(beta_vals, test_acc)

index = np.argmax(np.array(test_acc))
best_beta = beta_vals[index]
best_beta
max(test_acc)

num_batches = 3

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        offset = (step % num_batches * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels, beta: best_beta}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % nmm.accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % nmm.accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % nmm.accuracy(test_prediction.eval(), test_labels))


def NN_logits(train, weights, biases, prob=1.0):

    for i in range(len(weights)):
        logits = tf.matmul(train, weights[i]) + biases[i]
        train = tf.nn.relu(logits)
        train = tf.nn.dropout(train, prob)
    return logits


dimension_list = [image_size*image_size, hidden_nodes,
                  hidden_nodes//2, hidden_nodes//4, hidden_nodes//8, num_labels]
logit_maker = NN_logits

graph = tf.Graph()
with graph.as_default():
    def generate_variables(dimension_list):
        weights = []
        biases = []
        for i in range(1, len(dimension_list)):
            weights.append(tf.Variable(tf.truncated_normal(
                [dimension_list[i-1], dimension_list[i]], stddev=np.sqrt(2.0/dimension_list[i-1]))))
            biases.append(tf.Variable(tf.zeros([dimension_list[i]])))
        return weights, biases

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    beta = tf.placeholder(tf.float32)
    global_step = tf.Variable(0)

    # Variables.
    weights, biases = generate_variables(dimension_list)
    # Training computation.
    logits = logit_maker(tf_train_dataset,  weights, biases, 0.5)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    for weight in weights:
        loss = loss + beta*tf.nn.l2_loss(weight)

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.5, global_step, 5000, 0.8, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_logits = logit_maker(tf_valid_dataset, weights, biases)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_logits = logit_maker(tf_test_dataset, weights, biases)
    test_prediction = tf.nn.softmax(test_logits)

num_steps = 20001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels, beta: 1e-3}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % nmm.accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % nmm.accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % nmm.accuracy(test_prediction.eval(), test_labels))
