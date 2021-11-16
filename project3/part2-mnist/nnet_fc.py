#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32 # base 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(
              nn.Linear(784, 128),
              nn.ReLU(), # ReLU or LeakyReLU
              nn.Linear(128, 10),
            )
    lr=0.1 # base 0.1
    momentum=0 # base 0
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
   # loss, accuracy = run_epoch(test_batches, model.eval(), None)

   # print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
# Base Val: 0.932487, accuracy: 0.9204727564102564
# Batch: Val:  0.940020, Accuracy on test set: 0.9314903846153846
# Momentum: Val accuracy:   0.884358, Accuracy on test set: 0.8704927884615384
# Lr: Val accuracy:   0.934659, 0.9206730769230769
# LeakyReLU: Val accuracy:   0.931985 Accu: 0.9207732371794872


# 128 Neurons
# LeakyReLu Val accuracy:   0.978443 Accuracy on test set: 0.9772636217948718
# LR 0.01   Val accuracy:   0.955047 Accuracy on test set: 0.9427083333333334
# Momentum  Val accuracy:   0.965408 Accuracy on test set: 0.9606370192307693
# Batch 64  Val accuracy:   0.976310 Accuracy on test set: 0.9747596153846154