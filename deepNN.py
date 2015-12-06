import numpy as np
import math
import alcove
from global_utils import *

"""
A single hidden layer, deep neural network implementation
To be used as the I -> H -> O route in the 
memory network.
I'm not sure actually if we have a H layer.
Author: Jason Yim
"""



class deepNN:

    def __init__(self, input_size, hidden_size, 
                    output_size=0, hidden_layers = 1, 
                    l_rate = 0.1, activ_func= sigmoid):
        """
        args:
            input_size - size of input
            output_size - size of output
            hidden_size -  size of hidden_layers
                            for now each hidden layer has the same size
            hidden_layers - number of hidden layers
                            currently only working with 1
            l_rate - learning rate of network
        """

        if (output_size == 0):
            output_size = input_size

        self.l_rate = l_rate
        # making sure the layers are column vectors
        self.input = np.matrix(np.zeros(input_size)).T
        self.output = np.matrix(np.zeros(output_size)).T
        self.hidden = np.matrix(np.zeros([hidden_layers, input_size]))
        self.activ_func = activ_func

        # weight matrices
        self.W_hi = np.random.normal(loc=0,scale=1/np.sqrt(hidden_size),
                                    size=(hidden_size,input_size))
        self.B_h = np.random.normal(loc=0,scale=1,size=(hidden_size,1))
        self.W_oh = np.random.normal(loc=0,scale=1/np.sqrt(hidden_size),
                                    size=(output_size,hidden_size))
        self.B_o = np.random.normal(loc=0,scale=1,size=(output_size,1))


    def forward_pass(self, input):
        """
            Perform a forward pass on the network given
            an input vector.
        """
        if (input.shape != self.input.shape):
            print "input dimension does not match"

        self.input = input
        self.hidden = self.activ_func(np.dot(self.W_hi, self.input) + self.B_h)
        self.output = self.activ_func(np.dot(self.W_oh, self.hidden) + self.B_o)


    def backward_pass(self, target):
        """
            Perform a backward pass on the network given
            an output vector.
            returns the error
        """
        if (target.shape != self.output.shape):
            print "Target/output dimensions do not match"

        error = target - self.output

        # calculate deltas
        delta_o = np.multiply(np.multiply(error,self.output),1-self.output)
        delta_h = np.multiply(np.multiply(np.dot(self.W_oh.T, delta_o), self.output),
                                                                1-self.output)

        # deltas for weights
        W_oh_delta = np.dot(np.multiply(self.l_rate, delta_o), self.hidden.T)
        W_hi_delta = np.dot(np.multiply(self.l_rate, delta_h), self.input.T)

        # update weights
        self.W_hi = self.W_hi + W_hi_delta
        self.W_oh = self.W_oh + W_oh_delta

        return error

    def train(self, nepochs, print_error = True):
        """
            Performs training given input patterns ipats
            and test patterns tpats. "nepochs" denotes the number
            of epochs
        """
        size = len(ipats)
        terr = 0

        for (n in range(nepochs)):
            epocherr = 0
            for i in range(nepochs):
                ipat = ipats[i]
                tpat = tpats[i]
                forward_pass(ipat)
                error = backward_pass(tpat)
                epocherr = math.pow(np.matrix.sum(error),2)/2 + epocherr
            if print_error:
                print "Epoch #" + str(n+1) + "error: " + str(epocherr)
            terr = terr + epocherr
        if print_error:
            print "Total error: " + str(terr)


"""
def main():
    test = deepNN(3,3)
    ipat_1 = np.matrix(np.zeros(3)).T
    test.forward_pass(ipat_1)
    tpat_1 = np.matrix(np.zeros(3)).T
    test.backward_pass(tpat_1)
        

if __name__ == "__main__":
    main()
"""

