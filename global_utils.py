import numpy as np
from scipy.special import expit

"""Global variables and utility methods.

If they pop up frequently and are consistent across
different models then they should be stored here...
"""
INPUT_LENGTH = 4


def string_to_vector(s):
    """Convert a string to concatenated 1-of-26 vectors.

    -Jay
    """
    indices = [ord(x)-97 for x in s]
    vector = np.zeros(len(s)*26)
    vector[np.arange(0, 26*len(s), 26)+indices] = 1
    return vector


def sigmoid(x):
    """Sigmoid activation function. Works on scalars, vectors, matrices."""
    return expit(x)
    # return 1.0 / (1.0+np.exp(-1.0*x))


def logit(x):
    """Logit function (inverse of sigmoid).
    Works on scalars, vectors, matrices."""
    return np.log(x/(1-x))


def id(x):
    """Identity (linear) activatione."""
    return x


def cross_entropy(output, target):
    """Get cross entropy between ouput and desired target"""
    output = np.array(output)
    target = np.array(target)
    return -1*np.sum(target * np.log(output) + (1-target) * np.log(1-output))


def sum_of_squares_error(output, target):
    """Sum of squares error function"""
    return .5*np.sum(np.power(np.subtract(output,target),2))


def last_seen(m):
    """Grab the last memory.

    Since we store memories before the output processes them,
    we actually want to grab the second to last memory in the list.
    Since python lists are inherently circular, on the first run through,
    when there's only 1 memory, it will grab the memory of itself.
    """

    memory_weight = .1  # how much do we weight the last memory by?
    mem = m.memory_array[-2]  # grab second to last memory
    return mem*memory_weight


def store_every(I_activ, m, error=None):
    """Store any memory, no deletion or anything."""

    # make sure not going over memory limit
    # if so remove the oldest one

    if len(m.memory_array) == m.limit:
        m.memory_array = m.memory_array[1:]

    m.memory_array.append(I_activ)


def load_data(data_set):
    """Load data_set

    Args:
        data_set: file name (string)
    Returns:
        two lists of vectors: inputpatterns, outputpatterns
    """
    f = open(data_set)
    ipats = []
    tpats = []
    for line in f.readlines():
        items = line.split()
        ipat = string_to_vector(items[0])
        tup = items[-1]
        tpat = np.array([int(tup[1]), int(tup[3])])
        ipats.append(ipat)
        tpats.append(tpat)
    return ipats, tpats

if __name__ == "__main__":
    print string_to_vector("yoyo")
