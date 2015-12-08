import numpy as np
from global_utils import *

"""
A class version of the memory network.
Basically everything in network.py in
a class. This class deviates from the
memory network presented in the paper
to only satisfy the needs of our project

Author: Jason Yim, Jay Miller
"""

"""
NOTES:
1. Does the same activation function have to be
used at all steps? What about error function?

2. To test individual routes, set canonical_on,
memory_on to True/False
"""


class memory_network:

    def __init__(self, canonical_route, memory_route,
                 input_size, output_size, l_rate=.05, activ_func=sigmoid):
        self.canonical_route = canonical_route
        self.memory_route = memory_route
        self.output = np.matrix(np.zeros(output_size)).T
        self.input = np.matrix(np.zeros(input_size)).T
        self.B_o = np.random.normal(loc=0, scale=1, size=(output_size, 1))
        self.activ_func = activ_func
        self.l_rate = l_rate
        # turn off/on the canonical/memory routes
        self.canonical_on = False
        self.memory_on = True

    def forward_pass(self, input):
        if self.input.shape != input.shape:
            print "Input dimensions do not match"

        self.input = input
        if self.canonical_on:
            self.canonical_route.forward_pass(input)
            self.output = self.output + self.canonical_route.output
        if self.memory_on:
            self.memory_route.forward_pass(input)
            self.output = self.output + self.memory_route.a_out
        self.output = self.activ_func(self.output + self.B_o)
        self.canonical_route.output = self.output
        self.memory_route.a_out = self.output

    def backward_pass(self, target):
        if self.output.shape != target.shape:
            print "target/output dimensions do not match"
        error = sum_of_squares_error(self.output, target)
        dE_dOut = target - self.output
        dOut_dNet = np.subtract(np.ones((len(self.output), 1)), self.output)
        dOut_dNet = np.multiply(self.output, dOut_dNet)
        delta_out = np.multiply(dE_dOut, dOut_dNet)
        # update output bias
        self.B_o = self.B_o + self.l_rate*delta_out
        # backprop through both networks
        if self.canonical_on:
            self.canonical_route.backward_pass(dE_dOut)
        if self.memory_on:
            self.memory_route.backward_pass(dE_dOut)
        return error

    def train(self, ipats, tpats, nepochs, print_error=True):
        """ Performs training given input patterns ipat and
        test patterns tpat. "nepochs" denotes the number of epochs.
        """
        size = len(ipats)
        terr = 0
        for n in range(nepochs):
            epocherr = 0
            for i in range(size):
                ipat = ipats[i]
                tpat = tpats[i]
                self.forward_pass(ipat)
                error = self.backward_pass(tpat)
                epocherr = epocherr + error
            if print_error:
                print "Epoch #" + str(n+1) + " error: " + str(epocherr)
            terr = terr + epocherr
        if print_error:
            print "Total error: " + str(terr)
            return terr


""" Testing """
"""
def main():
    canonical = deepNN(3,3)
    memory = alcove(3,3,5)
    memory_net = memory_network(canonical,memory_network,3,3)

if __name__ == "__main__":
    main()

"""
