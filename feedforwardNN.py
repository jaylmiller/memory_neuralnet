import numpy as np
from global_utils import *


class DirectMappingNN:
    """
    No hidden layer, only input -> output.
    Asssuming, sigmoid activation on the output layer, although the sigmoid
    activation function for the output layer is not called inside of
    this network but instead inside of the memory network since the
    memory network computes the sigmoid of the sum of the net output
    of this network and alcove plus its bias vector.

    Author: Jason Yim, Jay Miller
    """
    def __init__(self, input_size,
                 output_size=0, l_rate=0.05):
        """
        args:
            input_size - size of input
            output_size - size of output
            l_rate - learning rate of network
        """

        if (output_size == 0):
            output_size = input_size

        self.l_rate = l_rate
        # making sure the layers are column vectors
        self.input = np.matrix(np.zeros(input_size)).T
        self.output = np.matrix(np.zeros(output_size)).T

        # weight matrices
        self.W_oi = np.random.normal(loc=0, scale=.5,
                                     size=(output_size, input_size))
        # no bias vector on output because it's in the memory_network

    def forward_pass(self, input):
        """ Perform a forward pass on the network given
        an input vector. Don't perform activation function
        on output, since the memory network does this itself on the
        sum of the two individual networks.
        """
        if (input.shape != self.input.shape):
            print "Input dimension does not match"

        self.input = input
        self.output = np.dot(self.W_oi, self.input)

    def backward_pass(self, dE_dOut):
        """ Perform a backward pass on the network given the
        negative of the derivative of the error with respect to the output.
        """
        if (dE_dOut.shape != self.output.shape):
            print "Target/output dimensions do not match"

        error = dE_dOut

        # calculate deltas
        delta_o = np.multiply(np.multiply(error, self.output), 1-self.output)

        # deltas for weights
        W_oi_delta = np.dot(np.multiply(self.l_rate, delta_o), self.input.T)
        self.W_oi = self.W_oi + W_oi_delta


class FeedForwardNN:
    """
    A single hidden layer, neural network implementation
    To be used as the I -> H -> O route in the
    memory network. Sigmoid activation on the hidden layer
    is assumed as is sigmoid activation on the output layer
    although the sigmoid activation function for the output layer
    is not called inside of this network but instead inside of the
    memory network since the memory network computes the sigmoid
    of the sum of the net output of this network and alcove plus
    its bias vector.

    Author: Jason Yim, Jay Miller
    """
    def __init__(self, input_size, hidden_size,
                 output_size=0, l_rate=0.05):
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
        self.hidden = np.matrix(np.zeros(hidden_size)).T
        self.activ_func = sigmoid

        # weight matrices
        self.W_hi = np.random.normal(loc=0, scale=.5,
                                     size=(hidden_size, input_size))
        self.B_h = np.random.normal(loc=0, scale=1, size=(hidden_size, 1))
        self.W_oh = np.random.normal(loc=0, scale=.5,
                                     size=(output_size, hidden_size))
        # no bias vector on output because it's in the memory_network

    def forward_pass(self, input):
        """ Perform a forward pass on the network given
        an input vector. Don't perform activation function
        on output, since the memory network does this itself on the
        sum of the two individual networks.
        """
        if (input.shape != self.input.shape):
            print "Input dimension does not match"

        self.input = input
        self.hidden = self.activ_func(np.dot(self.W_hi, self.input) + self.B_h)
        self.output = np.dot(self.W_oh, self.hidden)

    def backward_pass(self, dE_dOut):
        """ Perform a backward pass on the network given the
        negative of the derivative of the error with respect to the output.
        """
        if (dE_dOut.shape != self.output.shape):
            print "Target/output dimensions do not match"

        error = dE_dOut

        # calculate deltas
        delta_o = np.multiply(np.multiply(error, self.output), 1-self.output)
        delta_h = np.multiply(np.multiply(np.dot(self.W_oh.T, delta_o),
                                          self.hidden),
                              1-self.hidden)

        # deltas for weights
        W_oh_delta = np.dot(np.multiply(self.l_rate, delta_o), self.hidden.T)
        W_hi_delta = np.dot(np.multiply(self.l_rate, delta_h), self.input.T)

        # update weights and bias
        self.W_hi = self.W_hi + W_hi_delta
        self.W_oh = self.W_oh + W_oh_delta
        self.B_h = self.B_h + self.l_rate*delta_h
