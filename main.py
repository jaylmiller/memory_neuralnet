from memory_network import MemoryNetwork
from feedforwardNN import DirectMappingNN
from alcove import Alcove
import numpy as np
import random
from global_utils import load_phoneme_mapping
from matplotlib import pyplot as plt
from matplotlib import style


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
    plt.rcParams['toolbar'] = 'None'
    style.use('ggplot')
    random.seed(1)
    np.random.seed(1)
    load_phoneme_mapping()
    exemplar_nodes = 100

    ipats = load_data('datasets/ipat_484.txt', 'datasets/ipat_484_present.txt')
    tpats = load_data('datasets/tpat_484.txt', 'datasets/ipat_484_past.txt')

    ipats_binaries = ipats.values()
    tpats_binaries = tpats.values()

    input_size = len(ipats_binaries[0])
    output_size = len(tpats_binaries[0])

    canonical = DirectMappingNN(input_size,
                                output_size=output_size,
                                l_rate=.02)
    memory = Alcove(input_size, output_size, exemplar_nodes, r=2.0,
                    o_lrate=.02, a_lrate=.02)

    memory_net = MemoryNetwork(canonical, memory, input_size, output_size,
                               error_func="cross_entropy", l_rate=.02)
    # set both routes on
    MemoryNetwork.CANONICAL_ON = True
    MemoryNetwork.MEMORY_ON = True
    memory_net.train(ipats_binaries[:100], tpats_binaries[:100], 100)
    epe1 = memory_net.err_per_epoch
    plot_error_per_epoch([epe1], ['Dual route'], 'Average cross-entropy')


if __name__ == "__main__":
    main()
