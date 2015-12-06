import numpy as np
import math
import alcove
import deepNN
import sys

"""
A class version of the memory network.
Basically everything in network.py in 
a class.
Author: Jason Yim
"""


class memory_network:

    def __init__(self,canonical_route,memory_route,
        input_size,output_size):

        self.canonical_route = canonical_route
        self.memory_route = memory_route
        self.output = np.matrix(np.zeros(output_size)).T
        self.input = np,matrix(np.zeros(input_size)).T

    def forward_pass(self,input):
        if self.input.shape != input.shape:
            print "Input dimensions do not match"

        self.input = input
        self.canonical_route.forward_pass(input)
        self.memory_route.forward_pass(input)
        self.output = self.canonical_route.output + self.memory_route.output

    def backward_pass(self,target):
        if self.output.shape != target.shape:
            print "target/output dimensions do not match"

        error1 = self.canonical_route.backward_pass(target)
        error2 = self.memory_route.backward_pass(target)
        error = error1 + error2
        return error


    def train(self,ipats,tpats,print_error=True):
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
                forward_pass(ipat)
                error = backward_pass(tpat)
                # Using sum of squares error
                epocherr = math.pow(np.matrix.sum(error),2)/2 + epocherr
            if print_error
                print "Epoch #" + str(n+1) + " error: " + str(epocherr)
            terr = terr + epocherr
        if print_error
            print "Total error: " + str(terr)




def main():
    canonical = deepNN(3,3)
    memory = alcove(3,3,5)
    memory_net = memory_network(canonical,memory_network,3,3)

if __name__ == "__main__":
    main()




