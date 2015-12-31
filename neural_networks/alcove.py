"""Specialized implementation of the ALCOVE model (Kruschke 1992)
Intented to be used as a component of a memory_network object
(see memory_network.py). An activation function is not applied
to the output, since the parent memory_network does this, but it
is assumed by the backprop algorithm that the activation function
is logistic.

Authors: Jason Yim, Jay Miller
"""
import numpy as np
from global_utils import *
import random


class Alcove:

    def __init__(self, input_size, output_size, hidden_size,
                 spec=1.0, r=1.0, q=1.0, o_lrate=0.1, a_lrate=0.1, l_decay=.95):
        """
        args:
            input_size - size of input vectors
            output_size - size of output vectors
            hidden_size - size of hidden vectors
            spec - specificity of the node
            r,q - parameters of activation function
            o_lrate - learning rate for the output layer (association weights)
            a_lrate - learning rate for the hidden layer (attention weights)
        """
        self.param = (spec, r, q)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.o_lrate = o_lrate
        self.a_lrate = a_lrate
        self.l_decay = l_decay

        # Hidden layer, randomly sample with replacement from data
        # f = file('datasets/ipat_484.txt', 'r')
        # data_vectors = [np.matrix(map(int, line.rstrip().split(","))) for line in f]
        # data_vectors = data_vectors[:100]
        # random.shuffle(data_vectors)
        # self.node_vectors = np.vstack(tuple(data_vectors[:self.hidden_size]))
        # print np.vstack((data_vectors[0], data_vectors[1])).shape
        # self.node_vectors = data_vectors[:self.hidden_size]
        self.node_vectors = np.matrix(
            np.random.randint(2, size=(self.hidden_size, self.input_size)))
        self.node_vectors = self.node_vectors.astype(float)
        # Input nodes
        self.stimulus_nodes = np.matrix(np.zeros(self.input_size)).T
        # vector of "attention strengths"
        self.att_strengths = np.matrix(
            np.ones(self.input_size)).T
        # matrix of "association weights"
        self.assoc_weights = np.matrix(
            np.random.normal(loc=0,
                             scale=.5,
                             size=(self.output_size,
                                   self.hidden_size)))
        # activations
        self.a_in = np.matrix(np.zeros(self.input_size)).T
        self.a_hid = np.matrix(np.zeros(self.hidden_size)).T
        self.a_out = np.matrix(np.zeros(self.output_size)).T

    def forward_pass(self, input_vector):
        """ Perform a forward pass.

        input_vector should be a 1xn vector where n
        is the number of input nodes
        """
        self.a_in = input_vector
        self.hidden_activation_function()
        self.output_net()

    def backward_pass(self, dE_dOut, ep_num = None):
        """ Perform backward pass.

        "backward_pass" should be run in conjunction with "forward_pass"
        on the same intput-output pair.

        args:
            dE_dOut - the derivative of the error with respect to the output
                so we can plug in error deriv with entire network instead of
                just locally
        """

        delta_assoc = self.assoc_learn_sigmoid(dE_dOut)
        # delta_atten = self.atten_learn_sigmoid_jay(dE_dOut)
        delta_atten2 = self.atten_learn_sigmoid_jason(dE_dOut)
        # print np.max(np.abs(delta_atten2))
        # if (delta_atten == delta_atten2).all():
        #    print "equal"
        self.assoc_weights += delta_assoc
        self.att_strengths += delta_atten2
        # above_zeros = np.array(self.att_strengths > 0, dtype=int)
        # self.att_strengths = np.multiply(self.att_strengths, above_zeros)
        if ep_num is not None:
            if ep_num % 50 == 0:
                self.o_lrate = self.o_lrate*.95
                self.a_lrate = self.a_lrate*.95
        if self.o_lrate < .005:
            self.o_lrate = .005
            self.a_lrate = .005


    def assoc_learn_sigmoid(self, dE_dOut):
        """ Learn values for updating association weights
        """
        self.dOut_dNet = np.subtract(np.ones((len(self.a_out), 1)), self.a_out)
        self.dOut_dNet = np.multiply(self.a_out, self.dOut_dNet)
        delta = np.multiply(dE_dOut, self.dOut_dNet)
        return self.o_lrate*np.dot(delta, self.a_hid.T)

    def atten_learn_sigmoid_jason(self, dE_dOut):
        """ Jason's implementation of attention learning update
        (slightly updated by Jay, first few lines, to account for sigmoid act.)

        Compute delta for attention weights using the activation
        function described in the original ALCOVE paper
        """
        c, r, q = self.param
        # compute jacobian da_out/da_hid
        temp = self.dOut_dNet
        temp = np.squeeze(np.asarray(temp))
        da_out_da_hid = np.diag(temp)
        da_out_da_hid = np.dot(da_out_da_hid, self.assoc_weights)
        # compute each term separately for readability
        err_deriv = np.dot(dE_dOut.T, da_out_da_hid)
        sqr_diff_hid_in = np.power(np.subtract(self.node_vectors,
                                               self.a_in.T), r)
        net_hid = np.power(np.dot(sqr_diff_hid_in, self.att_strengths), 1/r)
        net_hid_pow = np.power(net_hid, q-r)
        # break this up into computing a few seperate variables then combine?
        # lots of computations at once, hard to see whats happening
        first_half = np.multiply(err_deriv.T,
                                 np.multiply(
                                     np.multiply(
                                         np.multiply(self.a_hid, c),
                                         q/r), net_hid_pow))
        return np.dot(sqr_diff_hid_in.T, first_half)

    def atten_learn_sigmoid_jay(self, dE_dOut):
        """ Learn values for updating attention strengths.
        Modification of derivation in the appendix.
        Computes dE/dalpha_i = (dE/da_out)(da_out/da_hid)(da_hid/dalpha_i)
        as in the appendix. (da_out/da_hid) differs because our output
        activation is nonlinear.
        """
        # compute jacobian da_out/da_hid
        temp = self.dOut_dNet
        temp = np.squeeze(np.asarray(temp))
        da_out_da_hid = np.diag(temp)
        da_out_da_hid = np.dot(da_out_da_hid, self.assoc_weights)
        # t1 is (dE/da_out)(da_out/da_hid)
        t1 = np.dot(dE_dOut.T, da_out_da_hid)
        # compute da_hid/dalpha_i
        c, r, q = self.param
        net_hid_pow = np.power(self.net_hid, q-r).T
        scalar = c*(q/r)
        temp = np.multiply(self.a_hid, net_hid_pow).T
        temp = scalar*temp
        delta_atten = []
        # compute elementwise like in appendix
        # (matrixify this later for more speed if needed)
        for i in range(self.input_size):
            node_act_norm_i = self.node_act_norm[i, :]
            dAhid_dAlpha = np.multiply(temp, node_act_norm_i)
            dot_prod = np.asscalar(np.dot(t1, dAhid_dAlpha.T))
            delta_atten.append(dot_prod)
        delta_atten = self.a_lrate*np.array(delta_atten)[:, np.newaxis]
        return delta_atten

    def error(self):
        """Calculates the sum of squares error (when teacher values used)"""
        return 0.5 * np.power(np.sum(np.subtract(self.t_val, self.a_out)), 2)

    def teacher_values(self, cout):
        """ Compute the teacher values
        given the correct output (cout)
        and the current activation of the
        output units.

        update:
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
        """
        c, r, q = self.param
        # separated the terms out to make it easier to read
        att_strengths = self.att_strengths
        self.node_act_norm = np.power(
            np.subtract(self.node_vectors, self.a_in.T), r).T
        self.net_hid = np.power(
            np.dot(att_strengths.T, self.node_act_norm), q/r)
        self.a_hid = np.exp(np.multiply(-c, self.net_hid)).T

    def output_net(self):
        """Set self.a_out to the net output so that
        it can be summed with the output of the other route
        and then have the activation function applied to their sum.
        """
        self.a_out = np.dot(self.assoc_weights, self.a_hid)
