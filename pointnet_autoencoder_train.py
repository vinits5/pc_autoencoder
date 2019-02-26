import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import helper
import transforms3d.euler as t3d
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='no_mode', help='mode: train or test')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_pose', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log_trial3', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float,default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--model_path', type=str, default='log_trial2/model200.ckpt', help='Path of the weights (.ckpt file) to be used for test')
parser.add_argument('--centroid_sub', type=bool, default=False, help='Centroid Subtraction from Source and Template before Pose Prediction.')
parser.add_argument('--use_pretrained_model', type=bool, default=False, help='Use a pretrained model of airplane to initialize the training.')
parser.add_argument('--use_random_poses', type=bool, default=False, help='Use of random poses to train the model in each batch')
parser.add_argument('--data_dict', type=str, default='templates',help='Data used to train templates or multi_model_templates')
parser.add_argument('--train_poses', type=str, default='itr_net_train_data.csv', help='Poses for training')
parser.add_argument('--eval_poses', type=str, default='itr_net_eval_data.csv', help='Poses for evaluation')
parser.add_argument('--feature_size', type=int, default=1024, help='Size of features extracted from PointNet')
FLAGS = parser.parse_args()


# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
	os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
	os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

TRAIN_POSES = FLAGS.train_poses
EVAL_POSES = FLAGS.eval_poses

# Change batch size during test mode.
if FLAGS.mode == 'test':
	BATCH_SIZE = 1
	FLAGS.model = 'pointnet_pose_test'
else:
	BATCH_SIZE = FLAGS.batch_size

# Parameters for data
NUM_POINT = FLAGS.num_point
MAX_NUM_POINT = 2048
NUM_CLASSES = 40
centroid_subtraction_switch = FLAGS.centroid_sub

# Network hyperparameters
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Model Import
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir

# Take backup of all files used to train the network with all the parameters.
if FLAGS.mode == 'train':
	if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)			# Create Log_dir to store the log.
	os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) 				# bkp of model def
	os.system('cp pointnet_autoencoder_train.py %s' % (LOG_DIR)) 	# bkp of train procedure
	os.system('cp -a utils/ %s/'%(LOG_DIR))						# Store the utils code.
	os.system('cp helper.py %s'%(LOG_DIR))
	LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')# Create a text file to store the loss function data.
	LOG_FOUT.write(str(FLAGS)+'\n')

# Write all the data of loss function during training.
def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)
 
# Calculate Learning Rate during training.
def get_learning_rate(batch):
	learning_rate = tf.train.exponential_decay(
						BASE_LEARNING_RATE,  # Base learning rate.
						batch * BATCH_SIZE,  # Current index into the dataset.
						DECAY_STEP,          # Decay step.
						DECAY_RATE,          # Decay rate.
						staircase=True)
	learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	return learning_rate        

# Get Batch Normalization decay.
def get_bn_decay(batch):
	bn_momentum = tf.train.exponential_decay(
					  BN_INIT_DECAY,
					  batch*BATCH_SIZE,
					  BN_DECAY_DECAY_STEP,
					  BN_DECAY_DECAY_RATE,
					  staircase=True)
	bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
	return bn_decay

def train():
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			batch = tf.Variable(0)									# That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.

		with tf.device('/gpu:'+str(GPU_INDEX)):
			is_training_pl = tf.placeholder(tf.bool, shape=())			# Flag for dropouts.
			bn_decay = get_bn_decay(batch)								# Calculate BN decay.
			learning_rate = get_learning_rate(batch)					# Calculate Learning Rate at each step.
			# Define a network to backpropagate the using final pose prediction.
			with tf.variable_scope('Network') as _:
				# Object of network class.
				network = MODEL.Network()
				# Get the placeholders.
				source_pointclouds_pl = network.placeholder_inputs(BATCH_SIZE, NUM_POINT)
				# Extract Features.
				source_global_feature = network.get_model(source_pointclouds_pl, FLAGS.feature_size, is_training_pl, bn_decay=bn_decay)
				# Find the predicted transformation.
				predicted_pointclouds_pl = network.decode_data(source_global_feature, is_training_pl, bn_decay=bn_decay)
				# Find the loss using source and transformed template point cloud.
				loss = network.get_loss_b(predicted_pointclouds_pl, source_pointclouds_pl)

			# Get training optimization algorithm.
			if OPTIMIZER == 'momentum':
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
			elif OPTIMIZER == 'adam':
				optimizer = tf.train.AdamOptimizer(learning_rate)

			# Update Network_L.
			train_op = optimizer.minimize(loss, global_step=batch)
			
		with tf.device('/cpu:0'):
			# Add the loss in tensorboard.
			tf.summary.scalar('learning_rate', learning_rate)
			tf.summary.scalar('bn_decay', bn_decay)						# Write BN decay in summary.
			saver = tf.train.Saver()
			tf.summary.scalar('loss', loss)

		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)

		# Add summary writers
		merged = tf.summary.merge_all()
		if FLAGS.mode == 'train':			# Create summary writers only for train mode.
			train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
									  sess.graph)
			eval_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval'))

		# Init variables
		init = tf.global_variables_initializer()
		sess.run(init, {is_training_pl: True})

		# Just to initialize weights with pretrained model.
		if FLAGS.use_pretrained_model:
			saver.restore(sess,os.path.join('log_trial1','model2000.ckpt'))

		# Create a dictionary to pass the tensors and placeholders in train and eval function for Network_L.
		ops = {'source_pointclouds_pl': source_pointclouds_pl,
			   'is_training_pl': is_training_pl,
			   'predicted_pointclouds_pl': predicted_pointclouds_pl,
			   'loss': loss,
			   'train_op': train_op,
			   'merged': merged,
			   'step': batch}

		if FLAGS.mode == 'train':
			# For actual training.
			for epoch in range(MAX_EPOCH):
				log_string('**** EPOCH %03d ****' % (epoch))
				sys.stdout.flush()
				# Train for all triaining poses.
				train_one_epoch(sess, ops, train_writer)
				# Save the variables to disk.
				saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
				if epoch % 50 == 0:
					# Evaluate the trained network after 50 epochs.
					eval_one_epoch(sess, ops, eval_writer)
					# Store the Trained weights in log directory.
				#if epoch % 50 == 0:
					save_path = saver.save(sess, os.path.join(LOG_DIR, "model"+str(epoch)+".ckpt"))
					log_string("Model saved in file: %s" % save_path)

		if FLAGS.mode == 'test':
			# Just to test the results
			test_one_epoch(sess, ops, saver, FLAGS.model_path)

		
# Train the Network_L and copy weights from Network_L to Network19 to find the poses between source and template.
def train_one_epoch(sess, ops, train_writer):
	# Arguments:
	# sess: 		Tensorflow session to handle tensors.
	# ops:		Dictionary for tensors of Network_L
	# ops19: 		Dictionary for tensors of Network19
	# templates:	Training Point Cloud data.
	# poses: 		Training pose data.

	is_training = True
	display_ptClouds = False

	train_file_idxs = np.arange(0, len(TRAIN_FILES))
	np.random.shuffle(train_file_idxs)

	for fn in range(len(TRAIN_FILES)):
		log_string('----' + str(fn) + '-----')
		current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
		current_data = current_data[:,0:NUM_POINT,:]
		current_data, _, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
		
		file_size = current_data.shape[0]
		num_batches = file_size // BATCH_SIZE
		loss_sum = 0

		# Training for each batch.
		for fn in range(num_batches):
			start_idx = fn*BATCH_SIZE 			# Start index of poses.
			end_idx = (fn+1)*BATCH_SIZE 		# End index of poses.
			
			template_data = current_data[start_idx:end_idx,:,:]	

			template_data = provider.rotate_point_cloud(template_data)
			template_data = provider.jitter_point_cloud(template_data)		

			# To visualize the source and point clouds:
			if display_ptClouds:
				helper.display_clouds_data(template_data[0])

			# Feed the placeholders of Network_L with source data and template data obtained from N-Iterations.
			feed_dict = {ops['source_pointclouds_pl']: template_data,
						 ops['is_training_pl']: is_training}

			# Ask the network to predict transformation, calculate loss using distance between actual points, calculate & apply gradients for Network_L and copy the weights to Network19.
			summary, step, _, loss_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss']], feed_dict=feed_dict)
			train_writer.add_summary(summary, step)		# Add all the summary to the tensorboard.

			# Display Loss Value.
			print("Batch: {} & Loss: {}\r".format(fn,loss_val)),
			sys.stdout.flush()

			# Add loss for each batch.
			loss_sum += loss_val
		
		log_string('Train Mean loss: %f' % (loss_sum/num_batches))		# Store and display mean loss of epoch.

def eval_one_epoch(sess, ops, eval_writer):
	# Arguments:
	# sess: 		Tensorflow session to handle tensors.
	# ops:		Dictionary for tensors of Network_L
	# ops19: 		Dictionary for tensors of Network19
	# templates:	Training Point Cloud data.
	# poses: 		Training pose data.

	is_training = False
	display_ptClouds = False

	test_file_idxs = np.arange(0, len(TEST_FILES))
	np.random.shuffle(test_file_idxs)

	for fn in range(len(TEST_FILES)):
		log_string('----' + str(fn) + '-----')
		current_data, current_label = provider.loadDataFile(TEST_FILES[test_file_idxs[fn]])
		current_data = current_data[:,0:NUM_POINT,:]
		current_data, _, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
		
		file_size = current_data.shape[0]
		num_batches = file_size // BATCH_SIZE
		loss_sum = 0											# Total Loss in each batch.
	
		for fn in range(num_batches):
			start_idx = fn*BATCH_SIZE 			# Start index of poses.
			end_idx = (fn+1)*BATCH_SIZE 		# End index of poses.
			
			template_data = current_data[start_idx:end_idx,:,:]

			# To visualize the source and point clouds:
			if display_ptClouds:
				helper.display_clouds_data(template_data[0])

			# Feed the placeholders of Network_L with source data and template data obtained from N-Iterations.
			feed_dict = {ops['source_pointclouds_pl']: template_data,
						 ops['is_training_pl']: is_training}

			# Ask the network to predict transformation, calculate loss using distance between actual points.
			summary, step, loss_val = sess.run([ops['merged'], ops['step'], ops['loss']], feed_dict=feed_dict)
			eval_writer.add_summary(summary, step)			# Add all the summary to the tensorboard.

			# Display Loss Value.
			print("Batch: {} & Loss: {}\r".format(fn,loss_val)),
			sys.stdout.flush()

			# Add loss for each batch.
			loss_sum += loss_val
		
		log_string('Eval Mean loss: %f' % (loss_sum/num_batches))		# Store and display mean loss of epoch.

def test_one_epoch(sess, ops, saver, model_path):
	# Arguments:
	# sess: 		Tensorflow session to handle tensors.
	# ops:		Dictionary for tensors of Network_L
	# ops19: 		Dictionary for tensors of Network19
	# templates:	Training Point Cloud data.
	# poses: 		Training pose data.
	# saver: 		To restore the weights.
	# model_path: 	Path of log directory.

	saver.restore(sess, model_path)			# Restore the weights of trained network.
	current_data, current_label = provider.loadDataFile(TEST_FILES[0])

	is_training = False
	display_ptClouds = False
		
	template_idx = 11
	current_data = current_data[template_idx*BATCH_SIZE:(template_idx+1)*BATCH_SIZE]
	print(current_data.shape)

	template_data = current_data[:,0:NUM_POINT,:]
	batch_euler_pose = np.array([[0,0,0,90*(np.pi/180), 0*(np.pi/180), 0*(np.pi/180)]])
	# template_data = helper.apply_transformation(template_data, batch_euler_pose)
	batch_euler_pose = np.array([[0,0,0,0*(np.pi/180), 0*(np.pi/180), 180*(np.pi/180)]])
	# template_data = helper.apply_transformation(template_data, batch_euler_pose)

	# To visualize the source and point clouds:
	if display_ptClouds:
		helper.display_clouds_data(template_data[0])

	# Feed the placeholders of Network_L with source data and template data obtained from N-Iterations.
	feed_dict = {ops['source_pointclouds_pl']: template_data,
				 ops['is_training_pl']: is_training}

	# Ask the network to predict transformation, calculate loss using distance between actual points.
	import time
	start = time.time()
	step, predicted_pointclouds_pl = sess.run([ ops['step'], ops['predicted_pointclouds_pl']], feed_dict=feed_dict)
	end = time.time()
	print('Time Elapsed: {}'.format(end-start))
	predicted_pointclouds_pl[0,:,0]=predicted_pointclouds_pl[0,:,0]+1
	helper.display_two_clouds(template_data[0], predicted_pointclouds_pl[0])


if __name__ == "__main__":
	if FLAGS.mode == 'no_mode':
		print('Specity a mode argument: train or test')
	elif FLAGS.mode == 'train':
		train()
		LOG_FOUT.close()
	else:
		train()