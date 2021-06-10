from keras.engine.topology import Layer
import keras.backend as K
import numpy as np


def activation_function(x):
	s_squared_norm = K.sum(K.square(x), axis=-1, keepdims=True)
	s_len = K.sqrt(s_squared_norm + K.epsilon())
	return x / s_len


class Routing(Layer):
	def __init__(self, routing_dim=512, routing_epoch=1, **kwargs):
		super(Routing, self).__init__(**kwargs)
		self.routing_dim = routing_dim
		self.routing_epoch = routing_epoch
		self.activation = activation_function

	def build(self, input_shape):
		super(Routing, self).build(input_shape)
		input_routing_dim = input_shape[-1]
		input_routing_num = input_shape[-2]
		self.W = self.add_weight(name='capsule_kernel', shape=(input_routing_num, input_routing_dim, self.routing_dim), initializer='glorot_uniform', trainable=True)

	def call(self, u_vecs):
		u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])
		batch_size = K.shape(u_vecs)[0]
		input_routing_num = K.shape(u_vecs)[1]
		u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_routing_num, 1, self.routing_dim))
		u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
		b = K.zeros_like(u_hat_vecs[:, :, :, 0])
		outputs = np.array([])
		for i in range(self.routing_epoch):
			b = K.permute_dimensions(b, (0, 2, 1))
			c = K.sigmoid(b)
			c = K.permute_dimensions(c, (0, 2, 1))
			b = K.permute_dimensions(b, (0, 2, 1))
			outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
			if i < self.routing_epoch - 1:
				b = K.batch_dot(outputs, u_hat_vecs, [2, 3])
		return outputs

	def compute_output_shape(self, input_shape):
		return (None, 1, self.routing_dim)


