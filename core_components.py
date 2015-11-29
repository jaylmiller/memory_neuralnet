"""Contains classes for core units of the memory network 

Reading the code with the diagram should clarify things. 
Each component is defined by a class, parameters for components
can be set by 
In this version of the model, the memory is implemented as a list
that can contain input patterns (in feature space) that are 
accesible by index.Of course, there are plently of other
possibilities, but this is what they use in the 
2015 Memory Network Paper written at Facebook.
-Jay
"""

import numpy as np
from global_utils import *


class RawInput:
    """See diagram: raw input component.
    
    This is the set of units whose values represent the raw
    training input pattern. Implemented as a vector
    which is the several 1-of-26 encodings concatenated in order
    where the first 1-of-26 encoding represents the first letter,
    and so on...  i.e. the vector will have dimension 26*INPUT_LENGTH
    -Jay
    """


    def __init__(self):
        self.units = np.zeros(INPUT_LENGTH*26)


    def clamp_input(self, string_vector):
        """Clamp an input pattern (in string form). 

        -Jay
        """

        self.units = string_vector
        return self.units


class I:
    """See diagram: input component. 

    I is a hidden layer that represents the input in a feature space.
    -Jay
    """


    def __init__(self, size, activ_func=sigmoid):
        self.size = size
        self.activation_val = np.zeros(size)
        self.activ_func = activ_func


class O:
    """See diagram: output component. 

    O is a hidden layer that represents output in a feature space (same as I).
    -Jay
    """

    def __init__(self, size, activ_func=sigmoid, memory_func=last_seen):
        self.size = size
        self.activation_val = np.zeros(size)
        self.activ_func = activ_func
        self.memory_func = memory_func


class G:
    """See diagram: generalizer component.

    This one takes inputs in feature space and applies a
    prescribed "generalizer function" onto M.
    -Jay
    """

    def __init__(self):
        """No generalizer given. Default to one which stores everything it can."""
        self.gen_func = store_every()

    def __init__(self, gen_func):
        """No generalizer given. Default to one which stores everything it can."""
        self.gen_func = gen_func

    def generalize(self, I_activ, memory_unit, error=None):
        """Take input from I_activ and using the generalizer function, alter memory"""
        self.gen_func(I_activ, memory_unit, error)


class M:
    """See diagram: memory component"""

    def __init__(self, limit=999999999999999999999):
        """If default limit value is used, this is unlimited memory"""
        self.limit = limit
        self.memory_array = []

class R:
    """Response component.

    Converts feature_space representation into response form 1x2 vector
    """
    def __init__(self, activ_func=sigmoid):
        self.units = np.zeros(2)
        self.activ_func = activ_func