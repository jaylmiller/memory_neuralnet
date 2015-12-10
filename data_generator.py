"""Creates the data sets needed to train/test our model

Author: Jay Miller, Jason Yim
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
    f.close()
    return freq_sequence

def get_regular_verbs(file_data,past=False):
    """ Creates a dictionary containing all regular verbs
    with their binary encodings

    args:
        file_data - file containing words and their encodings
        past -  if true then get the past tense of the regular verbs,
                if false then get present tense of regular verbs
    returns
        regular_verbs - dictionary containing orthographic and 
                        phonetic representation of the regulars
    """
    f = file('datasets/verblist_freqs.csv','r')
    regular_verbs = {}
    index_counter = 0
    for line in f:
        vals = [i.rstrip() for i in line.split(',')]
        if vals[-3] == "0":
            if past:
                regular_verbs[vals[1].lower()] = file_data[vals[1].lower()]
            else:
                regular_verbs[vals[0].lower()] = file_data[vals[0].lower()]
    f.close()
    return regular_verbs

def get_irregular_verbs(file_data,past=False):
    """ Creates a dictionary containing all irregular verbs
    with their binary encodings

    args:
        file_data - file containing words and their encodings
        past -  if true then get the past tense of the irregular verbs,
                if false then get present tense of irregular verbs
    returns
        irregular_verbs - dictionary containing orthographic and 
                        phonetic representation of the irregulars
    """
    f = file('datasets/verblist_freqs.csv','r')
    irregular_verbs = {}
    for line in f:
        vals = [i.rstrip() for i in line.split(',')]
        if vals[-3] == "1":
            if past:
                irregular_verbs[vals[1].lower()] = file_data[vals[1].lower()]
            else:
                irregular_verbs[vals[0].lower()] = file_data[vals[0].lower()]
    f.close()
    return irregular_verbs

def get_all_verbs(file_data,past=False):
    """ Creates a dictionary containing all verbs
    with their binary encodings

    args:
        file_data - file containing words and their encodings
        past -  if true then get the past tense of all verbs,
                if false then get present tense of all verbs
    returns
        irregular_verbs - dictionary containing orthographic and 
                        phonetic representation of all verbs
    """
    f = file('datasets/verblist_freqs.csv','r')
    all_verbs = {}
    for line in f:
        vals = [i.rstrip() for i in line.split(',')]
        if past:
            all_verbs[vals[1].lower()] = file_data[vals[1].lower()]
        else:
            all_verbs[vals[0].lower()] = file_data[vals[0].lower()]
    f.close()
    return all_verbs


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

def create_patterns(n, dist, file_data):
    """ Creates training patterns
    args:
        n - pattern size
        dist - distribution of words
        file_data - file data read in containing words ith vector encodings

    return:
        ipats - a dictionary containing the words as keys and vector encodings as values
    """
    indices = get_indices_from_dist(n,dist);
    pat = {}
    keys = file_data.keys()
    for index in indices:
        pat[keys[index]] = file_data[keys[index]]
    return pat

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








