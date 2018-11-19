from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import importlib
import random
import hashlib
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

import notMNIST_methods as nmm
importlib.reload(nmm)
# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline


def disp_samples(data_folders, sample_size):
    for folder in data_folders:
        print(folder)
        image_files = os.listdir(folder)
        image_sample = random.sample(image_files, sample_size)
        for image in image_sample:
            image_file = os.path.join(folder, image)
            i = Image(filename=image_file)
            display(i)


def disp_samples_pickle(pickle_file, sample_size=8):
    plt.figure()
    folder = ''.join(pickle_file)[-8]
    plt.suptitle(folder)
    try:
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print('Unable to read data from', pickle_file, ':', e)
        raise
    print("Number of images in ", folder, ":", len(dataset))
    for i, img in enumerate(random.sample(list(dataset), sample_size)):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.imshow(img)


def disp_samples_dataset(dataset, labels):
    pretty_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    plt.figure()
    items = random.sample(range(len(labels)), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.title(pretty_labels[labels[item]])
        plt.imshow(dataset[item], cmap='gray')


def uniqueness(A, B, labels_A):
    hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in A])
    hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in B])
    overlap = []  # list of indexes
    for i, hash1 in enumerate(hash_1):
        duplicates = np.where(hash_2 == hash1)
        if len(duplicates[0]):
            overlap.append(i)
    return np.delete(A, overlap, 0), np.delete(labels_A, overlap, None), overlap


url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'  # Change me to store data elsewhere

train_filename = nmm.maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = nmm.maybe_download('notMNIST_small.tar.gz', 8458043)


num_classes = 10
np.random.seed(133)


train_folders = nmm.maybe_extract(train_filename)
test_folders = nmm.maybe_extract(test_filename)

disp_samples(train_folders, 3)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

train_datasets = nmm.maybe_pickle(train_folders, 45000)
test_datasets = nmm.maybe_pickle(test_folders, 1800)

for pickle_file in train_datasets:
    disp_samples_pickle(pickle_file)

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = nmm.merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = nmm.merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = nmm.randomize(train_dataset, train_labels)
test_dataset, test_labels = nmm.randomize(test_dataset, test_labels)
valid_dataset, valid_labels = nmm.randomize(valid_dataset, valid_labels)

disp_samples_dataset(train_dataset, train_labels)
disp_samples_dataset(test_dataset, test_labels)
disp_samples_dataset(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

valid_dataset_u, valid_labels_u, overlap = uniqueness(
    valid_dataset[:200], train_dataset, valid_labels[:200])
#valid_dataset_u, valid_labels_u = uniqueness(valid_dataset, train_dataset, valid_labels)
# THIS TAKES FOREVER
# Logistic Regression Time!!!

test_dataset.flatten()

logistic_model = LogisticRegression()
x_test = test_dataset.reshape(test_dataset.shape[0], image_size**2)
y_test = test_labels

sample_size = 1000
x_train = train_dataset[:sample_size].reshape(sample_size, image_size**2)
y_train = train_labels[:sample_size]
logistic_model.fit(x_train, y_train)
logistic_model.score(x_test, y_test)

pred_labels = logistic_model.predict(x_test)
disp_samples_dataset(test_dataset, pred_labels)
big_logistic_model = LogisticRegression(solver='sag')
x_test = test_dataset.reshape(test_dataset.shape[0], image_size**2)
y_test = test_labels
x_train = train_dataset[:].reshape(len(train_dataset), image_size**2)
y_train = train_labels[:]
x_valid = valid_dataset.reshape(len(valid_dataset), image_size**2)
y_valid = valid_labels

big_logistic_model.fit(x_train, y_train)
big_logistic_model.score(x_test, y_test)
pred_labels = big_logistic_model.predict(x_test)
disp_samples_dataset(test_dataset, pred_labels)
big_logistic_model.score(x_valid, y_valid)
logistic_model.score(x_valid, y_valid)
