from memory_network import memory_network
from feedforwardNN import feedforwardNN
from alcove import alcove
import numpy as np


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


def main():
    # np.seterr(invalid='raise')
    hidden_layer_size = 10
    exemplar_nodes = 200

    ipats = load_data('datasets/ipat.txt', 'datasets/ipat_present.txt')
    tpats = load_data('datasets/tpat.txt', 'datasets/ipat_past.txt')

    ipats_binaries = ipats.values()
    tpats_binaries = tpats.values()

    input_size = len(ipats_binaries[0])
    output_size = len(tpats_binaries[0])

    canonical = feedforwardNN(input_size, hidden_layer_size, output_size=output_size)
    memory = alcove(input_size, output_size, exemplar_nodes, r=2)

    memory_net = memory_network(canonical, memory, input_size, output_size)
    memory_net.train(ipats_binaries, tpats_binaries, 200)

if __name__ == "__main__":
    main()
