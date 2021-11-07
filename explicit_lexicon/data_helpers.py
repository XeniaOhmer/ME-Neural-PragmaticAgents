import numpy as np


def generate_inputs(size, n, k, distribution):
    """ Generates the message input for the listener. The occurrence frequencies of the n messages follow either
        a uniform or a Zipfian distribution.
        inputs:
            size - number of data points
            n - number of states and messages in the system
            k - number of frequency ranks (from rank 1) that are available
            distribution - the different input distributions
        outputs:
            data - training data
            labels - training labels
    """

    data = np.zeros((size, n, n))

    if distribution == 'uniform':
        selection = np.random.choice(k, size=(size))
    elif distribution == 'Zipf':
        norm_factor = np.sum([1 / i for i in range(1, k + 1)])
        selection = np.random.choice(np.arange(k), size=size, p=[(1 / i) / norm_factor for i in range(1, k + 1)])

    for i in range(k):
        data[selection == i, i, :] = 1.
    labels = np.zeros((size, n))
    labels[np.arange(size), selection] = 1
    return np.float32(data), selection, labels


def generate_referential_inputs_more_distractors(n, n_reference_games):

    norm_factor = np.sum([1/i for i in range(1, n+1)])
    norm_factor2 = np.sum([(1/i)/norm_factor for i in range(1,n)])
    selections = []
    for i in range(n_reference_games):
        selection = np.random.choice(np.arange(n-1),
                                     size=3,
                                     p=[(1/i) / norm_factor for i in range(1, n)] / norm_factor2,
                                     replace=False)
        selections.append(selection)
    return np.array(selections)
