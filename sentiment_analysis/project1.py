from string import punctuation, digits
import numpy as np
import random

import utils
# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    loss = label * (np.dot(theta, feature_vector) + theta_0)
    hloss = np.amax([0, 1 - loss])
    return hloss
    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    egs = feature_matrix.shape[0]
    hloss_egs = np.zeros((egs))
    for i in range(0, feature_matrix.shape[0], 1):
        loss = 1 - labels[i] * (np.dot(theta, feature_matrix[i]) + theta_0)
        hloss = np.amax([0, loss])
        hloss_egs[i] = hloss
    hloss_full = (1 / egs) * np.sum(hloss_egs)
    return hloss_full
    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    eps = 1.e-7
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= eps:
        update_theta = label * feature_vector
        update_theta_0 = label * 1
        new_theta = (current_theta + update_theta, current_theta_0 + update_theta_0)
    else:
        new_theta = (current_theta, current_theta_0)
    return new_theta

    raise NotImplementedError


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            new_theta = perceptron_single_step_update(feature_matrix[i],
                        labels[i],
                        theta,
                        theta_0)[0]
            new_theta_0 = perceptron_single_step_update(feature_matrix[i],
                        labels[i],
                        theta,
                        theta_0)[1]
            theta = new_theta
            theta_0 = new_theta_0
            pass
    new_theta = (theta, theta_0)
    return new_theta
    raise NotImplementedError


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    sample_size = 0
    theta = np.zeros(feature_matrix.shape[1])
    sum_theta = theta
    theta_0 = 0
    sum_theta_0 = theta_0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            sample_size = sample_size + 1
            new_theta = perceptron_single_step_update(feature_matrix[i],
                        labels[i],
                        theta,
                        theta_0)[0]
            new_theta_0 = perceptron_single_step_update(feature_matrix[i],
                                                      labels[i],
                                                      theta,
                                                      theta_0)[1]
            sum_theta = sum_theta + new_theta
            sum_theta_0 = sum_theta_0 + new_theta_0
            theta = new_theta
            theta_0 = new_theta_0
    mean_theta = (1 / sample_size) * sum_theta
    mean_theta_0 = (1 / sample_size) * sum_theta_0
    theta_vec = (mean_theta, mean_theta_0)
    return theta_vec
    raise NotImplementedError


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    eps = 1.e-7
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) < 1:
        current_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        current_theta_0 = current_theta_0 + eta * label
    elif np.abs(label * (np.dot(current_theta, feature_vector) + current_theta_0) - 1) < eps:
        current_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        current_theta_0 = current_theta_0 + eta * label
    else:
        current_theta = (1 - eta * L) * current_theta
    theta_tuple = (current_theta, current_theta_0)
    return theta_tuple
    raise NotImplementedError


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0
    counter = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            counter = counter + 1
            eta = 1 / np.sqrt(counter)
            new_theta = pegasos_single_step_update(
                    feature_matrix[i],
                    labels[i],
                    L,
                    eta,
                    current_theta,
                    current_theta_0)[0]
            new_theta_0 = pegasos_single_step_update(
                    feature_matrix[i],
                    labels[i],
                    L,
                    eta,
                    current_theta,
                    current_theta_0)[1]
            current_theta = new_theta
            current_theta_0 = new_theta_0
    theta = (current_theta, current_theta_0)
    return theta
    raise NotImplementedError

# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    eps = 1.e-7
    obs = feature_matrix.shape[0]
    lf = np.zeros(obs)
    cls = np.zeros(obs)
    for i in range(obs):
        lf = np.dot(theta, feature_matrix[i]) + theta_0
        if lf >= eps:
            cls[i] = 1
        else:
            cls[i] = -1
    return cls
    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    theta_trained = classifier(train_feature_matrix, train_labels, **kwargs)[0]
    theta_0_trained = classifier(train_feature_matrix, train_labels, **kwargs)[1]
    pred_trained = classify(train_feature_matrix, theta_trained, theta_0_trained)
    accu_train = accuracy(pred_trained, train_labels)
    pred_val = classify(val_feature_matrix, theta_trained, theta_0_trained)
    accu_val = accuracy(pred_val, val_labels)
    tuple = (accu_train, accu_val)
    return tuple
    raise NotImplementedError


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

stopwords = utils.load_data('stopwords.txt')
print(f'length: {len(stopwords)}')
stop_keys = stopwords
# stop_keys = str.split(stopwords, ",")

my_file = open("stopwords.txt", "r")
content = my_file.read()
# print(content)

content_list = content.split("\n")
my_file.close()
print(content_list)
print(type(content_list))
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    dictionary = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
#        for word in word_list:
#            if word in content_list:
#                 del word_list.index(word)
        for word in word_list:
            if word not in content_list:
                if word not in dictionary:
                    dictionary[word] = len(dictionary)
    return dictionary
# print(f'\ntype of stopwords: {stopwords}\n')
# print(f'\ntype of stop keys: {stop_keys}\n')



def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()


