from core_components import *
from global_utils import *
from autoencoder import *
import numpy as np
import math


"""Here we can define a network by combining the various components

In other words construct each component as on object from its respective class in 
the core_components module. (Check the constructor methods to see how this is done)

Also weight matrices and bias vectors should be addded here. As well as a training
algorithm, and testing algorithm.
"""

feature_dim = 50 # fairly arbitrary for now
memory_space = 50 # fairly arbitrary for now
I_lrate = 0.1 # arbitrary

ipats, tpats = load_data('dataset1.txt')

def get_exact_or_zero(m, feature_vector):
    """Given a memory component m, return feature_vector if in memory, otherwise
    return the 0 vector. 
    """
    for vector in m.memory_array:
        if (feature_vector==vector).all():
            return vector
    return np.zeros(26*INPUT_LENGTH)

def generalizer_function(I_active, m, error=None):
    store_every(I_active, m) # place holder


raw_in = RawInput()
i_comp = I(feature_dim) # use default sigmoid activation
o_comp = O(feature_dim, memory_func=get_exact_or_zero) # use default sigmoid activation
g_comp = G(generalizer_function)
m_comp = M(memory_space)
r_comp = R()


"""Weights initialized with Gaussian, mean = 0, var = 1/num_inputs
Biases initialized with Gaussian, mean = 0, var = 1
See http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
"""
W_iraw = np.random.normal(loc=0, scale=1/np.sqrt(feature_dim), 
                          size=(feature_dim,INPUT_LENGTH*26))
B_i = np.random.normal(loc=0, scale=1, size=(feature_dim, 1))
W_oi = np.random.normal(loc=0, scale=1/np.sqrt(feature_dim),
                         size=(feature_dim,feature_dim))
B_o = np.random.normal(loc=0, scale=1, size=(feature_dim, 1))
W_om = np.random.normal(loc=0, scale=1/np.sqrt(feature_dim), 
                         size=(feature_dim,feature_dim))
W_ro = np.random.normal(loc=0, scale=1/np.sqrt(feature_dim), 
                         size=(2,feature_dim))
B_r = np.random.normal(loc=0, scale=1, size=(2, 1))

"""
Pretrain the I layer as an autoencoder... I think pretraining
this is justifiable because there would already be some internal
representation of words learned by the child. We are keeping their
internal representation fixed by pretraining it, and then not effecting
it when it is learning the verbs

This is necessary because otherwise the memory layer won't work..... 
"""
a = AutoEncoder(W_iraw, B_i)
W_iraw, B_i = a.train(ipats)

def forward_pass(ipat):
    if ipat.size != INPUT_LENGTH*26:
        return
    # I layer
    raw_in.clamp_input(ipat)
    i_comp.units = np.dot(W_iraw, np.matrix(raw_in.units).T)+B_i
    i_comp.units = i_comp.activ_func(i_comp.units)
    # generalizer
    generalizer_function(i_comp.units, m_comp)
    # i to o
    o_comp.units = np.dot(W_oi, i_comp.units)
    # get memory
    mem_vec = o_comp.memory_func(m_comp, i_comp.units)
    # memory to o
    o_comp.units = o_comp.units+np.dot(W_om, mem_vec)+B_o
    # apply activation
    o_comp.units = o_comp.activ_func(o_comp.units)
    # o to r
    r_comp.units = np.dot(W_ro, o_comp.units)+B_r
    r_comp.units = r_comp.activ_func(r_comp.units)
    #print r_comp.units


def backward_pass(tpat,error):
    """ Perorm backpropagation
    """

    global W_oi
    global W_ro


    tpat = np.matrix(tpat).T
    
    # calculate deltas
    delta_r = np.multiply(np.multiply(error, r_comp.units), 1-r_comp.units)
    delta_o = np.multiply(np.multiply(np.dot(np.transpose(W_ro), delta_r), o_comp.units),(1-o_comp.units))

    # deltas for weights
    W_delta_ro = np.dot(np.multiply(I_lrate, delta_r), o_comp.units.T)
    W_delta_oi = np.dot(np.multiply(I_lrate, delta_o), i_comp.units.T)

    # update weights
    W_oi = W_oi + W_delta_oi
    W_ro = W_ro + W_delta_ro


def train(ipat, tpat, nepochs):
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
            # reshape tpat
            tpat = np.matrix(tpat).T
            forward_pass(ipat)
            error = tpat - r_comp.units
            # Using sum of squares error
            epocherr = math.pow(np.matrix.sum(error),2)/2 + epocherr
            backward_pass(tpat,error)
        print "Epoch #" + str(n+1) + " error: " + str(epocherr)
        terr = terr + epocherr
    print "Total error: " + str(terr)

def main():
    train(ipats,tpats, 50)
        

if __name__ == "__main__":
    main()


