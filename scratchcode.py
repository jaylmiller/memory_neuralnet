def atten_learn_linear(self, dE_dOut):
    """ Simplified code for Jason's version. They output the same thing.
    """
    c, r, q = self.param
    net_hid_pow = np.power(self.net_hid, q-r).T
    scalar = c*(q/r)
    dE_dOut_x_dOut_dHid = np.dot(dE_dOut.T, self.assoc_weights)
    temp = np.multiply(self.a_hid, net_hid_pow).T
    temp = scalar*temp
    dAhid_dAlpha = np.dot(temp, self.node_act_norm.T)
    return np.dot(dE_dOut_x_dOut_dHid, dAhid_dAlpha)

def atten_learn(self, dE_dOut):
    """ Compute delta for attention weights using the activation
    function described in the original ALCOVE paper
    """
    c, r, q = self.param

    # compute each term separately for readability
    err_deriv = np.dot(dE_dOut.T, self.assoc_weights)
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

def assoc_learn_linear(self, dE_dOut):
    """ Compute delta for association weights with linear activation
    """
    return self.o_lrate*np.dot(dE_dOut, self.a_hid.T)