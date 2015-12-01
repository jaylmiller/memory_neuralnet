""" General Case Implementation of the ALCOVE model (Kruschke 1992)
Author: Jason Yim
"""
import numpy as np
import math


"""
TO DO:
1. change all ndarrays into matrices at very beginning
"""


class Alcove:

	def __init__(self, input_size, output_size, hidden_size, 
			spec, r, q, o_lrate = 0.1, a_lrate = 0.1):
		"""
		args:
			input_size - size of input vectors
			output_size - size of output vectors
			hidden_size - size of hidden vectors
			spec - specificity of the node
			r,q - parameters of activation function
		"""
		self.param = (spec,r,q)
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.o_lrate = o_lrate
		self.a_lrate = a_lrate

		# Hidden layer
		self.node_vectors = np.random.normal(loc=0, scale=1/np.sqrt(self.hidden_size),
											size=(self.hidden_size,self.input_size))
		# Input nodes
		self.stimulus_nodes = np.zeros(self.input_size)
		# vector of "attention strengths"
		self.att_strengths = np.zeros(self.input_size)
		# matrix of "association weights"
		self.assoc_weights = np.zeros((self.output_size, self.hidden_size))
		# activations
		self.a_in = np.zeros(self.input_size)
		self.a_hid = np.zeros(self.hidden_size)
		self.a_out = np.zeros(self.output_size)



	def forward_pass(self, input_vector):
		""" Perform a forward pass.
		input_vector should be a 1xn vector where n 
		is the number of input nodes
		
		"""
		self.a_in = input_vector
		self.hidden_activation_function()
		self.category_activation_function()

	def backward_pass(self, correct_output):
		""" Perform backward pass.
		"correct_output" is the correct category for the input_vector
		"backward_pass" should be run in conjunction with "forward_pass"
		on the same intput-output pair.
		"""
		self.t_val = self.teacher_values(correct_output)
		error = self.error()
		delta_assoc = self.assoc_learn()
		
	def assoc_learn(self):
		# convert ndarrays into matrices
		t_val = np.matrix(self.t_val)
		a_out = np.matrix(self.a_out)
		


		return np.dot(np.multiply(self.o_lrate, np.subtract(t_val, a_out)),
					self.a_hid)



	def error(self):
		""" Calculates the sum of squares error """
		return 0.5 * np.power(np.sum(np.subtract(self.t_val ,self.a_out)), 2)

	def teacher_values(self, cout):
		""" Compute the teacher values
		given the correct output (cout) 
		and the current activation of the
		output units
		"""
		a_c_out = np.append(np.matrix(self.a_out),cout,axis=0)
		w,l = a_c_out.shape
		t_values = np.zeros(l)
		ones_bit_mask = np.ones(2).T
		zeros_bit_mask = np.zeros(2).T

		for i in range(l):
			col = a_c_out[:,i]
			if np.array_equal(np.subtract(col,ones_bit_mask), zeros_bit_mask):
				t_values[i] = max(col[0], 1)
			else:
				t_values[i] = min(col[0], -1)
		return t_values


	def hidden_activation_function(self):
		""" Compute the activation of hidden layer (exemplar nodes)
		Refer to Equaion (1) in Kruschke
		"""
		c,r,q = self.param
		# separated the terms out to make it easier to read
		att_strengths = self.att_strengths.T
		hidd_layer_minus_a_in_sqred = np.power(np.subtract(self.node_vectors, self.a_in), r).T
		dot_prod_sqrt = np.power(np.dot(att_strengths, hidd_layer_minus_a_in_sqred), float(q)/r)

		self.a_hid = np.exp(np.multiply(-c,  dot_prod_sqrt)).T

	def category_activation_function(self):
		self.a_out = np.dot(self.a_hid, self.assoc_weights.T)

def main():
	""" Testing """

	test = Alcove(5, 2, 10, 1, 2, 1)
	input_vector = np.zeros(5).T
	correct_output = np.matrix([1, 0])
	test.forward_pass(input_vector)
	test.backward_pass(correct_output)

if __name__ == "__main__":
	main()