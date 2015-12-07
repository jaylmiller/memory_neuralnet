"""Specialized implementation of the ALCOVE model (Kruschke 1992)
Authors: Jason Yim, Jay Miller
"""
import numpy as np
import math
from global_utils import *


class alcove:

    def __init__(self, input_size, output_size, hidden_size,
                 spec=1, r=1, q=1, o_lrate=0.1, a_lrate=0.1):
        """
        args:
            input_size - size of input vectors
            output_size - size of output vectors
            hidden_size - size of hidden vectors
            spec - specificity of the node
            r,q - parameters of activation function
        """
        self.param = (spec, r, q)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.o_lrate = o_lrate
        self.a_lrate = a_lrate

        # Hidden layer
        self.node_vectors = np.matrix(np.random.normal(loc=0,
                                                       scale=1 /
                                                       np.sqrt(
                                                           self.hidden_size),
                                                       size=(self.hidden_size,
                                                             self.input_size)))
        # Input nodes
        self.stimulus_nodes = np.matrix(np.zeros(self.input_size)).T
        # vector of "attention strengths"
        #self.att_strengths = np.matrix(np.zeros(self.input_size)).T
        self.att_strengths = np.matrix(np.random.normal(loc=0,scale=1,size=(self.input_size))).T
        # matrix of "association weights"
        #self.assoc_weights = np.matrix(np.zeros((self.output_size, self.hidden_size)))
        self.assoc_weights = np.matrix(np.random.normal(loc=0,scale=1,size=(self.output_size, self.hidden_size)))
        # activations
        self.a_in = np.matrix(np.zeros(self.input_size)).T
        self.a_hid = np.matrix(np.zeros(self.hidden_size)).T
        self.a_out = np.matrix(np.zeros(self.output_size)).T
        # save net activation of hidden layer during forward pass
        # for computing gradients during backprop
        self.net_hid = np.matrix(np.zeros(self.hidden_size))

    def forward_pass(self, input_vector):
        """ Perform a forward pass.

        input_vector should be a 1xn vector where n
        is the number of input nodes
        """
        self.a_in = input_vector
        self.hidden_activation_function()
        self.output_activation_function()

    def backward_pass(self, target_vector):
        """ Perform backward pass.

        "backward_pass" should be run in conjunction with "forward_pass"
        on the same intput-output pair.
        UPDATE:
        Changing the learning rule since we're using sigmoid on the
        output now. -Jay

        args:
            target_vector - correct vector

        returns:
            the error

        """

        error = sum_of_squares_error(self.a_out, target_vector)

        """ Had to switch, assoc_learn_sigmoid was not 
        outputting correct weight matrix """
        delta_assoc = self.assoc_learn_linear(target_vector)
        delta_atten = self.atten_learn(target_vector)
        self.assoc_weights += delta_assoc
        self.att_strengths += delta_atten
        return error

    def assoc_learn_linear(self, target_vector):
        """ Compute delta for association weights with linear activation
        """
        return np.dot(np.multiply(self.o_lrate, np.subtract(target_vector,
                                                            self.a_out)),
                      self.a_hid.T)

    def assoc_learn_sigmoid(self, target_vector):
        """ Added by Jay:
        Compute delta for association weights with sigmoid activation
        """
        t = np.subtract(np.ones((len(self.a_out),1)), self.a_out)
        self.gradient_out = np.multiply(self.a_out, t)
        print self.a_out.shape
        print t.shape
        print self.gradient_out.shape
        diff = np.subtract(target_vector, self.a_out)
        print diff.shape
        return self.o_lrate*np.multiply(self.gradient_out, diff)

    def atten_learn(self, target_vector):
        """ Compute delta for attention weights using the activation
        function described in the original ALCOVE paper
        """
        c, r, q = self.param

        # compute each term separately for readability
        err_deriv = np.dot(np.subtract(target_vector, self.a_out).T, self.assoc_weights)
        sqr_diff_hid_in = np.power(np.subtract(self.node_vectors,
                                               self.a_in.T), r)
        net_hid = np.power(np.dot(sqr_diff_hid_in,self.att_strengths), 1/r)
        net_hid_pow = np.power(net_hid, q-r)
        # break this up into computing a few seperate variables then combine?
        # lots of computations at once, hard to see whats happening
        first_half = np.multiply(err_deriv.T,
                                 np.multiply(
                                     np.multiply(
                                         np.multiply(self.a_hid, c),
                                         q/r), net_hid_pow))
        return np.dot(sqr_diff_hid_in.T,first_half)

    def atten_learn_v2(self, target_vector):
        """ Simplified and more efficient code than original atten_learn.
        This uses values computed during the forward pass
        instead of recomputing them during the backprop like the other
        method did.

        -Jay
        """
        c, r, q = self.param
        net_hid_pow = np.power(self.net_hid, q-r)
        print net_hid_pow.shape
        print self.node_act_norm.shape
        print self.a_hid.shape
        scalar = c*(q/r)
        gradient_hid = np.dot(self.node_act_norm, net_hid_pow.T)

    def error(self):
        """Calculates the sum of squares error (when teacher values used)"""
        return 0.5 * np.power(np.sum(np.subtract(self.t_val, self.a_out)), 2)

    def teacher_values(self, cout):
        """ Compute the teacher values
        given the correct output (cout)
        and the current activation of the
        output units.

        UPDATE:
        I'm not sure if teacher values are still appropriate so
        I'll leave this method in here just in case.
        -Jay
        """
        a_c_out = np.append(self.a_out, cout, axis=0)
        w, l = a_c_out.shape
        t_values = np.matrix(np.zeros(l))
        ones_bit_mask = np.ones(2).T
        zeros_bit_mask = np.zeros(2).T
        for i in range(l):
            col = a_c_out[:, i]
            if np.array_equal(np.subtract(col, ones_bit_mask), zeros_bit_mask):
                t_values[0, i] = max(col[0], 1)
            else:
                t_values[0, i] = min(col[0], -1)
        return t_values

    def hidden_activation_function(self):
        """ Compute the activation of hidden layer (exemplar nodes)
        Refer to Equaion (1) in Kruschke.

        Note to Jason from Jay:
            If you store some of the terms computed during
            the forward pass here, backprop code will be much
            easier/efficient/readable
        """
        c, r, q = self.param
        # separated the terms out to make it easier to read
        att_strengths = self.att_strengths

        self.node_act_norm = np.power(
            np.subtract(self.node_vectors, self.a_in.T), r).T
        self.net_hid = np.power(
            np.dot(att_strengths.T, self.node_act_norm), q/r)
        self.a_hid = np.exp(np.multiply(-c, self.net_hid)).T

    def output_activation_function(self):
        """Changing this to sigmoid activation. -Jay
        """
        self.a_out = np.dot(self.assoc_weights, self.a_hid)
        self.a_out = sigmoid(self.a_out)

    def train(self,nepochs,ipats,tpats,print_error=True):
        size = len(ipats)
        terr = 0
        for n in range(nepochs):
            epocherr = 0
            for i in range(size):
                ipat = ipats[i]
                tpat = tpats[i]
                self.forward_pass(ipat)
                error = self.backward_pass(tpat)
                epocherr = error + epocherr
            if print_error:
                print "Epoch #" + str(n+1) + " error: " + str(epocherr)
            terr = terr + epocherr
        if print_error:
            print "Total error: " + str(terr)















