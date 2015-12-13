from memory_network import MemoryNetwork
from feedforwardNN import DirectMappingNN, FeedForwardNN
from alcove import Alcove
import numpy as np
import random
from global_utils import load_phoneme_mapping, save_net, load_net
from matplotlib import pyplot as plt
from matplotlib import style
from data_generator import *




def plot_error_per_epoch(err_per_epochs, legend_list, error_type, interval=1):
    """
    args:
        err_per_epochs - should be a list of the different lines to plot
        legend_list - list of strings naming each line
        error_type - string containing type of error
    """
    plt.figure(facecolor='gray')
    x = np.linspace(0, 350, len(err_per_epochs[0]))
    for err in err_per_epochs:
        plt.plot(x, err)
    plt.xlabel("Epoch number")
    plt.ylabel(error_type)
    # plt.xlim(1, x[-1])
    plt.ylim(0, 1)
    plt.legend(legend_list)
    plt.show()


def train_net():
    """Train the network.
    """
    # Set parameters
    # plt.rcParams['toolbar'] = 'None'
    style.use('ggplot')
    random.seed(1)
    np.random.seed(1)
    exemplar_nodes = 100
    training_size = 100
    epochs = 500

    phoneme_mapping = load_phoneme_mapping()

    ipats, tpats = load_ipat_tpat('datasets/ipat_484.txt',
                                  'datasets/tpat_484.txt')

    pat_labels = load_labels()
    distribution = create_distribution()
    random_indices = get_indices_from_dist(training_size, distribution)
    ipats_train = [ipats[i] for i in random_indices]
    tpats_train = [tpats[i] for i in random_indices]
    input_size = ipats_train[0].shape[0]
    output_size = tpats_train[0].shape[0]

    canonical = DirectMappingNN(input_size, output_size=output_size,
                                l_rate=.02)
    memory = Alcove(input_size, output_size, exemplar_nodes, r=2.0,
                    o_lrate=.1, a_lrate=.1, l_decay=.95)

    memory_net = MemoryNetwork(canonical, memory, input_size, output_size,
                               error_func="cross_entropy", l_rate=.02)
    # set both routes on
    # MemoryNetwork.CANONICAL_ON = False
    # MemoryNetwork.MEMORY_ON = False
    memory_net.train(ipats_train, tpats_train, epochs)
    epe1 = memory_net.err_per_epoch
    plot_error_per_epoch([epe1], ['Memory route'], 'Average cross-entropy')

""" net.train(ipats_binaries, tpats_binaries, 1)
    t = 0
    for i in range(input_size):
        ipat = ipats_binaries[i]
        tpat = tpats_binaries[i]
        t = t + test_accuracy(net, ipat, tpat, phoneme_mapping)
    print t
    print "accuracy: " + str(float(t)/float(input_size)) """

def run_tests(data_folder='', num_epochs=1000, interval=50):
    """Run tests on a network that is loaded from saved data.
    """


    # plt.rcParams['toolbar'] = 'None'
    style.use('ggplot')
    random.seed(1)
    np.random.seed(1)
    training_size = 100

    phoneme_mapping = load_phoneme_mapping()

    ipats, tpats = load_ipat_tpat('datasets/ipat_484.txt',
                                  'datasets/tpat_484.txt')

    pat_labels = load_labels()
    distribution = create_distribution()
    random_indices = get_indices_from_dist(training_size, distribution)
    ipats_train = [ipats[i] for i in random_indices]
    tpats_train = [tpats[i] for i in random_indices]
    ipats_reg = []
    tpats_reg = []
    ipats_ireg = []
    tpats_ireg = []

    for i in random_indices:
        if pat_labels[i] == 1:
            ipats_ireg.append(ipats[i])
            tpats_ireg.append(tpats[i])
        elif pat_labels[i] == 0:
            ipats_reg.append(ipats[i])
            tpats_reg.append(tpats[i])

    acc_all = [0]
    acc_reg = [0]
    acc_ireg = [0]
    print "computing accuracy"
    for i in range(interval, num_epochs+1, interval):
        fname = data_folder + 'net_at_' + str(i)
        net = load_net(fname)
        # MemoryNetwork.MEMORY_ON = False
        # MemoryNetwork.CANONICAL_ON = False
        acc_all.append(get_mean_accuracy(net, ipats_train, tpats_train, phoneme_mapping))
        print acc_all[-1]
        acc_reg.append(get_mean_accuracy(net, ipats_reg, tpats_reg, phoneme_mapping))
        acc_ireg.append(get_mean_accuracy(net, ipats_ireg, tpats_ireg, phoneme_mapping))
    acc_all = np.array(acc_all)
    acc_reg = np.array(acc_reg)
    acc_ireg = np.array(acc_ireg)
    plot_error_per_epoch([acc_all, acc_reg, acc_ireg], ['All patterns', 'Regular patterns', 'Irregular patterns'], 'Accuracy', interval=50)

def run_tests2(data_folder='', num_epochs=1000, interval=50):
    """Run tests on a network that is loaded from saved data.
    """


    # plt.rcParams['toolbar'] = 'None'
    style.use('ggplot')
    random.seed(1)
    np.random.seed(1)
    training_size = 1000

    phoneme_mapping = load_phoneme_mapping()

    ipats, tpats = load_ipat_tpat('datasets/ipat_484.txt',
                                  'datasets/tpat_484.txt')

    pat_labels = load_labels()
    distribution = create_distribution()
    random_indices = get_indices_from_dist(training_size, distribution)
    ipats_train = [ipats[i] for i in random_indices]
    tpats_train = [tpats[i] for i in random_indices]
    ipats_reg = []
    tpats_reg = []
    ipats_ireg = []
    tpats_ireg = []

    for i in random_indices:
        if pat_labels[i] == 1:
            ipats_ireg.append(ipats[i])
            tpats_ireg.append(tpats[i])
        elif pat_labels[i] == 0:
            ipats_reg.append(ipats[i])
            tpats_reg.append(tpats[i])

    net = load_net('net_at_1000')
    MemoryNetwork.CANONICAL_ON = False
    memory_net_in = []
    for i in range(len(ipats_train)):
        net.memory_route.forward_pass(ipats_train[i])
        memory_net_in.append(net.memory_route.a_out)

    print np.mean(memory_net_in)
    print np.std(memory_net_in)

if __name__ == "__main__":
    # train_net()
    run_tests(data_folder='alcove_', num_epochs = 500)
