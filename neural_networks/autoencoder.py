"""Train an autoencoder to be used as a hidden layer in the network.

-Jay
"""

import numpy as np
from global_utils import *
import random


class AutoEncoder:

    def __init__(self, weights, bias, activation_func=sigmoid):
        """Initializer.

        Args:
            weights: initial weight matrix that will be updated by training
                the autoencoder
            bias: initial bias vector that will be updated
            activation_func: activation function on hidden units
        """

        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    def train(self, data, epochs=300, learnrate=.1):
        """Train the autoencoder.

        Args:
            data: Data to use
            epochs: epochs to run
            learnrate: learning-rate
        Returns:
            tuple: new weights and bias
        """
        for e in range(epochs):
            random.shuffle(data)
            total_ce = 0
            for data_vector in data:
                # if data wrong size skip it
                if data_vector.size != INPUT_LENGTH*26:
                    continue
                e = self.encode(data_vector)  # get encoding
                d = self.decode(e)  # get decoding
                c_e = cross_entropy(d, data_vector)  # compute cross-entropy
                total_ce = total_ce + c_e

                # compute deltas

                delta_out = np.matrix(data_vector).T-d
                if self.activation_func == sigmoid:
                    delta_hidden = np.dot(self.weights, delta_out)
                    delta_hidden = np.multiply(delta_hidden, e)
                    delta_hidden = np.multiply(delta_hidden, 1-e)
                else:
                    print "don't know derivative of activation function"
                    return

                # delta_Woh = learnrate * np.dot(delta_out, e.T)
                delta_Whi = learnrate * \
                    np.dot(delta_hidden, np.matrix(data_vector))
                delta_Bh = learnrate * delta_hidden

                # update weights and bias

                self.weights = self.weights + delta_Whi
                self.bias = self.bias + delta_Bh

        return (self.weights, self.bias)

    def encode(self, data_vector):
        """Encode a single data_vector
        """
        net = np.dot(self.weights, np.matrix(data_vector).T)+self.bias
        return self.activation_func(net)

    def decode(self, encoded_vector):
        """Decode an encoded_vector
        """
        net = np.dot(self.weights.T, encoded_vector)
        return self.activation_func(net)


if __name__ == "__main__":
    ipats, tpats = load_data('dataset1.txt')
    feature_dim = 30
    W_iraw = np.random.normal(loc=0, scale=1/np.sqrt(feature_dim),
                              size=(feature_dim, INPUT_LENGTH*26))
    B_i = np.random.normal(loc=0, scale=1, size=(feature_dim, 1))
    a = AutoEncoder(W_iraw, B_i)
    a.train(ipats)
