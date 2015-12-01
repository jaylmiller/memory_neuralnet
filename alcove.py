"""Implementation of the ALCOVE model (Kruschke 1992)
"""
import numpy as np


class Alcove:

	def __init__(self, input_size, output_size, node_vectors):
		"""
		args:
			input_size - size of input vectors
			output_size - size of output vectors
			node_vectors - a list of vectors representing nodes in hidden layer
		"""
		self.num_nodes = len(node_vectors)
		self.node_vectors = node_vectors
		# matrix of "attention strengths"
		self.att_strengths = np.zeros((self.num_nodes, self.input_size))
		# matrix of "association weights"
		self.assoc_weights = np.zeros((self.output_size, self.num_nodes))
		# activations
		self.a_in = np.zeros(self.input_size)
		self.a_hid = np.zeros(self.num_nodes)
		self.a_out = np.zeros(self.a_out)


	def forward_pass(self, input_vector):
		self.a_in = input_vector
		


	def activation_function(self):


	def 

