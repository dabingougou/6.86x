import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        self.input_dimension = 5120 # 2560 # 42 rows * 28 cols
        self.conv1 = nn.Conv2d(1, 32, (3, 3)) # 40 rows * 26 cols
        self.pool1 = nn.MaxPool2d((2, 2)) # 20 rows * 13 cols
        self.conv2 = nn.Conv2d(32, 64, (3, 2)) # 18 rows * 12 cols
        self.pool2 = nn.MaxPool2d((2, 2))   # 9 rows and 6 cols
        self.conv3 = nn.Conv2d(64, 128, (2, 2)) # 8 rows and 5 cols
        self.flat = nn.Flatten()    # 8 * 5 * 128 = 5120
#        self.flat = nn.Flatten()    # 20 * 13 * 32 = 8320
        self.drop = nn.Dropout(p=0.5)
        self.out  = nn.Linear(input_dimension, 20)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.out(x)
        out_first_digit = x[:, 0:10]
        out_second_digit = x[:, 10:]
        # TODO use model layers to predict the two digits

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
