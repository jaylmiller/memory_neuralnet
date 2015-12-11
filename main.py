from memory_network import MemoryNetwork
from feedforwardNN import DirectMappingNN
from alcove import Alcove
import numpy as np
import random
from global_utils import load_phoneme_mapping, save_net, load_net
from matplotlib import pyplot as plt
from matplotlib import style
from data_generator import *




def plot_error_per_epoch(err_per_epochs, legend_list, error_type):
    """
    args:
        err_per_epochs - should be a list of the different lines to plot
        legend_list - list of strings naming each line
        error_type - string containing type of error
    """
    plt.figure(facecolor='gray')
    x = np.array([i for i in range(1, len(err_per_epochs[0])+1)])
    for err in err_per_epochs:
        plt.plot(x, err)
    plt.xlabel("Epoch number")
    plt.ylabel(error_type)
    plt.xlim(1, x[-1])
    plt.legend(legend_list, loc='upper right')
    plt.show()


def main():
    # Set parameters
    plt.rcParams['toolbar'] = 'None'
    style.use('ggplot')
    random.seed(1)
    np.random.seed(1)
    exemplar_nodes = 500
    training_size = 1000
    epochs = 1000
    benchmark_per = 50 # test at every benchmark_per

    phoneme_mapping = load_phoneme_mapping()


    # read files
    present_tense_words = load_data('datasets/ipat_484.txt', 'datasets/ipat_484_present.txt')
    past_tense_words = load_data('datasets/tpat_484.txt', 'datasets/ipat_484_past.txt')

    # get word distribution based on frequency
    distribution = create_distribution()

    # get testing sets of regular, irregular verbs
    regular_ipat = get_regular_verbs(present_tense_words)
    regular_tpat = get_regular_verbs(past_tense_words, past=True)
    irregular_ipat = get_irregular_verbs(present_tense_words)
    irregular_tpat = get_irregular_verbs(past_tense_words,past=True)
    # get testing sets of all verbs
    all_verbs_ipat = get_all_verbs(present_tense_words)
    all_verbs_tpat = get_all_verbs(past_tense_words,past=True)

    [ipats,tpats] = create_patterns(training_size, distribution, present_tense_words, past_tense_words)

    ipats_binaries = ipats.values()
    tpats_binaries = tpats.values()

    input_size = len(ipats_binaries[0])
    output_size = len(tpats_binaries[0])

    canonical = DirectMappingNN(input_size,
                                output_size=output_size,
                                l_rate=.1)
    memory = Alcove(input_size, output_size, exemplar_nodes, r=2.0,
                    o_lrate=.2, a_lrate=.1)

    memory_net = MemoryNetwork(canonical, memory, input_size, output_size,
                               error_func="cross_entropy", l_rate=.1)
    # set both routes on
    MemoryNetwork.CANONICAL_ON = True
    MemoryNetwork.MEMORY_ON = True
    memory_net.train(ipats_binaries, tpats_binaries, epochs)
    epe1 = memory_net.err_per_epoch
    plot_error_per_epoch([epe1], ['Dual route'], 'Average cross-entropy')



if __name__ == "__main__":
    main()
