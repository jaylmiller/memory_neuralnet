"""Creates the data sets needed to train/test our model

Author: Jay Miller, Jason Yim
"""

import numpy as np


def create_distribution(sample_indices=None):
    """ Create the probability distribution.

    returns:
        an array of numbers from 0 to 1, representing
        the distribution
    """

    f = file('datasets/verblist_freqs.csv', 'r')
    freq_sequence = [0]
    total_freq = 0
    for idx, line in enumerate(f):
        if sample_indices is not None and (idx == sample_indices).any():
            continue
        vals = [i.rstrip() for i in line.split(',')]
        freq = float(vals[-2])
        freq_sequence.append(freq_sequence[-1]+freq)
        total_freq = total_freq + freq
    freq_sequence = np.array(freq_sequence)
    # normalize
    freq_sequence = freq_sequence/freq_sequence[-1]
    f.close()
    return freq_sequence

def get_indices_from_dist(n, dist, sample_indices=None):
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
        if sample_indices is None:
            indices.append(idx - 1)
        else:
            indices.append(sample_indices[idx - 1])
    return indices


def load_ipat_tpat(binary_file_i, binary_file_t):
    fi = open(str(binary_file_i), 'r')
    ft = open(str(binary_file_t), 'r')
    ipats = []
    tpats = []
    for l in fi:
        v = np.matrix(map(int, l.rstrip().split(","))).T
        ipats.append(v)
    for l in ft:
        v = np.matrix(map(int, l.rstrip().split(","))).T
        tpats.append(v)

    return (ipats, tpats)


def load_labels():
    f = file('datasets/verblist_freqs.csv', 'r')
    labels_list = []
    for l in f:
        vals = [i.rstrip() for i in l.split(',')]
        labels_list.append(int(vals[-3]))
    return labels_list


def test_accuracy(net, ipat, tpat, phoneme_mapping):
    """Get accuracy for single ipat-tpat pair
    """
    phonemes = net.predict_phonemes(ipat, phoneme_mapping)
    print phonemes
    tpat_matrix = np.reshape(tpat, (10, 16))
    output_string = ""
    for i in range(10):
        v = tpat_matrix[i, :]
        for key in phoneme_mapping:
            if (phoneme_mapping[key] == v).all():
                output_string = output_string + str(key)
    if phonemes == output_string:
        return 1
    else:
        return 0

def test_accuracy_reg(net, ipat, tpat, phoneme_mapping):
    """Get accuracy for single ipat-tpat pair
    """
    phonemes = net.predict_phonemes(ipat, phoneme_mapping)
    # print phonemes
    tpat_matrix = np.reshape(tpat, (10, 16))
    output_string = ""
    for i in range(10):
        v = tpat_matrix[i, :]
        for key in phoneme_mapping:
            if (phoneme_mapping[key] == v).all():
                output_string = output_string + str(key)
    if phonemes[-2:] == output_string[-2:]:
        return 1
    else:
        return 0


def get_mean_accuracy(net, ipats_binaries, tpats_binaries, phoneme_mapping):
    t = 0
    input_size = len(ipats_binaries)
    for i in range(input_size):
        ipat = ipats_binaries[i]
        tpat = tpats_binaries[i]
        t = t + test_accuracy_reg(net, ipat, tpat, phoneme_mapping)
    return float(t)/float(input_size)