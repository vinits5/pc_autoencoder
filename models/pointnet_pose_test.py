import tensorflow as tf
import numpy as np
import math
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
# from transform_nets import input_transform_net, feature_transform_net
# import tf_util_loss

class Network:
	def placeholder_inputs(self,batch_size, num_point):
		# with tf.variable_scope('inputs') as ip:
		source_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
		return source_pointclouds_pl

	def get_model(self, source_pointclouds_pl, feature_size, is_training, bn_decay=None):
		""" Classification PointNet, input is BxNx3, output Bx40 """
		# with tf.variable_scope('PointNet') as pn:

		# Comment above two lines to have same points for loss and features and also change the variable names in the next line.
		batch_size = source_pointclouds_pl.get_shape()[0].value
		num_point = source_pointclouds_pl.get_shape()[1].value
		end_points = {}

		input_image = tf.expand_dims(source_pointclouds_pl, -1)

		net = tf_util.conv2d(input_image, 128, [1,3],
							 padding='VALID', stride=[1,1],
							 bn=True, is_training=is_training,
							 scope='conv1', bn_decay=bn_decay)

		net = tf_util.conv2d(net, 256, [1,1],
							 padding='VALID', stride=[1,1],
							 bn=True, is_training=is_training,
							 scope='conv2', bn_decay=bn_decay, activation_fn=None)

#		net = tf_util.conv2d(net, 64, [1,1],
#							 padding='VALID', stride=[1,1],
#							 bn=True, is_training=is_training,
#							 scope='conv3', bn_decay=bn_decay)
#		net = tf_util.conv2d(net, 128, [1,1],
#							 padding='VALID', stride=[1,1],
#							 bn=True, is_training=is_training,
#							 scope='conv4', bn_decay=bn_decay)

#		net = tf_util.conv2d(net, 1024, [1,1],
#							 padding='VALID', stride=[1,1],
#							 bn=True, is_training=is_training,
#							 scope='conv5', bn_decay=bn_decay)

		# Symmetric function: max pooling
		source_feature = tf_util.max_pool2d(net, [num_point, 1],
								 padding='VALID', scope='maxpool')
		source_feature = tf.tile(source_feature, [1, num_point, 1, 1])
		source_feature = tf.concat([net, source_feature], axis=3)
		
		net = tf_util.conv2d(source_feature, 512, [1,1],
		 					 padding='VALID', stride=[1,1],
							 bn=True, is_training=is_training,
							 scope='conv3', bn_decay=bn_decay)

		net = tf_util.conv2d(net, 1024, [1,1],
		 					 padding='VALID', stride=[1,1],
		 					 bn=True, is_training=is_training,
		 					 scope='conv4', bn_decay=bn_decay, activation_fn=None)
		source_global_feature = tf_util.max_pool2d(net, [num_point, 1],
		 						 padding='VALID', scope='maxpool')
		source_global_feature = tf.reshape(source_global_feature, [batch_size, -1])

		return source_global_feature

	def decode_data(self, source_global_feature, is_training, bn_decay=None):
		batch_size = source_global_feature.get_shape()[0].value
		net = tf_util.fully_connected(source_global_feature, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
		net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)		
		net = tf_util.fully_connected(net, 1024*3, activation_fn=None, scope='fc3')
		predicted_pointclouds_pl = tf.reshape(net, [batch_size, 1024, 3])
		return predicted_pointclouds_pl

	def get_loss_b(self, predicted_pointclouds_pl, source_pointclouds_pl):
		# def get_loss_b(self, predicted_pointclouds_pl, source_pointclouds_pl):
		# with tf.variable_scope('loss') as LossEvaluation:
			# loss = tf.reduce_mean(tf.square(tf.subtract(predicted_pointclouds_pl, source_pointclouds_pl)))
			# loss = tf_util_loss.chamfer(predicted_pointclouds_pl, source_pointclouds_pl)
		loss = 0
		return loss

if __name__=='__main__':
	with tf.Graph().as_default():
		net = Network()
		inputs = tf.zeros((1,1024,3))
		outputs = net.get_model(inputs, 1024, tf.constant(True))
		pt_cloud = net.decode_data(outputs, tf.constant(True))
		loss = net.get_loss_b(pt_cloud, inputs)