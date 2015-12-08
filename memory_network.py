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
For testing individual routes
"""


class MemoryNetwork:
    CANONICAL_ON = True
    MEMORY_ON = True

    def __init__(self, canonical_route, memory_route,
                 input_size, output_size, l_rate=.05, activ_func=sigmoid):
        self.canonical_route = canonical_route
        self.memory_route = memory_route
        self.output = np.matrix(np.zeros(output_size)).T
        self.input = np.matrix(np.zeros(input_size)).T
        self.B_o = np.random.normal(loc=0, scale=1, size=(output_size, 1))
        self.activ_func = activ_func
        self.l_rate = l_rate

    def forward_pass(self, input):
        """Forward pass through network.
        """
        if self.input.shape != input.shape:
            print "Input dimensions do not match"

        self.input = input
        if MemoryNetwork.CANONICAL_ON:
            self.canonical_route.forward_pass(input)
            self.output = self.output + self.canonical_route.output
        if MemoryNetwork.MEMORY_ON:
            self.memory_route.forward_pass(input)
            self.output = self.output + self.memory_route.a_out
        self.output = self.activ_func(self.output + self.B_o)
        # clamp the outputs of the two networks to the output of the
        # parent network
        self.canonical_route.output = self.output
        self.memory_route.a_out = self.output

    def backward_pass(self, target):
        """Backward pass. Update bias vector for output,
        call backpropogation on the individual networks.

        Returns:
            error (sum of squares)
        """
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
        if MemoryNetwork.CANONICAL_ON:
            self.canonical_route.backward_pass(dE_dOut)
        if MemoryNetwork.MEMORY_ON:
            self.memory_route.backward_pass(dE_dOut)
        return error

    def train(self, ipats, tpats, nepochs, print_error=True):
        """ Performs training given input patterns ipat and
        test patterns tpat. "nepochs" denotes the number of epochs.

        TODO: Add graphing functionality and other experimental features.
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
