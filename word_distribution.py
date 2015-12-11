"""For sampling words according to their frequency data

Author: Jay Miller
"""

import numpy as np


def create_distribution():
    """ Create the probability distribution.

    returns:
        an array of numbers from 0 to 1, representing
        the distribution
    """

    f = file('datasets/verblist_freqs.csv', 'r')
    freq_sequence = [0]
    total_freq = 0
    for line in f:
        vals = [i.rstrip() for i in line.split(',')]
        freq = float(vals[-2])
        freq_sequence.append(freq_sequence[-1]+freq)
        total_freq = total_freq + freq
    freq_sequence = np.array(freq_sequence)
    # normalize
    freq_sequence = freq_sequence/freq_sequence[-1]
    return freq_sequence


def get_indices_from_dist(n, dist):
    """ Get randomly sampled indices according to distribution

    args:
        n - number of indices desired
        dist - the distribution (computed from create_distribution)
    """
    indices = []
    for i in range(n):
        r = np.random.rand()
        condition = (dist <= r)
        idx = np.count_nonzero(condition)
        indices.append(idx - 1)
    return indices

def create_patterns(indices,words):
    ipat = {}
    keys = words.keys()
    for index in indices:
        ipat[keys[index]] = words[keys[index]]
    return ipat

