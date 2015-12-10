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

def load_data(binary, ortho):
    """ Load dataset
    binary - binary vector representation of verbs
    ortho - the words themselves

    return:
    ipat - a dictionary with the words as keys and binary vectors as values
    """
    bin_repres = open(binary)
    orth_repres = open(ortho)
    num_lines = sum(1 for line in orth_repres)  # get number of lines

    orth_repres.close()  # highly inefficient, any other way?
    orth_repres = open(ortho)

    ipats = {}
    for line in range(num_lines):
        bin = np.matrix(map(int, bin_repres.readline().rstrip().split(","))).T
        orth = orth_repres.readline().rstrip()
        ipats[orth] = bin
    return ipats
