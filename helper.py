import csv
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import transforms3d.euler as t3d
import helper
import tensorflow as tf


###################### Data Handling Operations #########################

# Read the templates from a given file.
def read_templates(file_name,templates_dict):
	with open(os.path.join('data',templates_dict,file_name),'r') as csvfile:
		csvreader = csv.reader(csvfile)
		data = []
		for row in csvreader:
			row = [float(i) for i in row]
			data.append(row)
	return data 										# n2 x 2048 x 3

# Read the file names having templates.
def template_files(templates_dict):
	with open(os.path.join('data',templates_dict,'template_filenames.txt'),'r') as file:
		files = file.readlines()
	files = [x.strip() for x in files]
	print('Templates used to train data: ')
	print(files)
	return files 										# 1 x n1

# Read the templates from each file.
def templates_data(templates_dict):
	files = template_files(templates_dict)							# Read the available file names.
	data = []
	for i in range(len(files)):
		temp = read_templates(files[i],templates_dict)
		for i in temp:
			data.append(i)
	return np.asarray(data)								# (n1 x n2 x 2048 x 3) & n = n1 x n2

# Preprocess the templates and rearrange them.
def process_templates(templates_dict):
	data = templates_data(templates_dict)								# Read all the templates.
	print('No. of Total Templates: {}'.format(data.shape[0]/2048))
	templates = []
	for i in range(data.shape[0]/2048):
		start_idx = i*2048
		end_idx = (i+1)*2048
		templates.append(data[start_idx:end_idx,:])
	return np.asarray(templates)						# Return all the templates (n x 2048 x 3)

# Read poses from given file.
def read_poses(templates_dict, filename):
	# Arguments:
		# filename:			Read data from a given file (string)
	# Output:
		# poses:			Return array of all the poses in the file (n x 6)

	with open(os.path.join('data',templates_dict,filename),'r') as csvfile:
		csvreader = csv.reader(csvfile)
		poses = []
		for row in csvreader:
			row = [float(i) for i in row]
			poses.append(row)
	return np.asarray(poses)

# To Store the features obtained from network.
def store_features(filename,feature):
	# Arguments:
		# filename: 		name of file to store features.
		# feature:			Array of feature to store in the file.
	with open(filename,'w') as file:
		for i in range(1024):
			file.write(str(feature[i]))
			file.write(',')


###################### Transformation Operations #########################

def rotate_point_cloud_by_angle_y(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		#rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]])
		shape_pc = batch_data[k, ...]
		# rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
		rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape((-1, 3)).T).T 		# Pre-Multiplication (changes done)
	return rotated_data

def rotate_point_cloud_by_angle_x(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		#rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[1, 0, 0],
									[0, cosval, -sinval],
									[0, sinval, cosval]])
		shape_pc = batch_data[k, ...]
		# rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
		rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape((-1, 3)).T).T 		# Pre-Multiplication (changes done)
	return rotated_data

def rotate_point_cloud_by_angle_z(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		#rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, -sinval, 0],
									[sinval, cosval, 0],
									[0, 0, 1]])
		shape_pc = batch_data[k, ...]
		# rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
		rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape((-1, 3)).T).T 		# Pre-Multiplication (changes done)
	return rotated_data

# Translate the data as per given translation vector.
def translate(data,shift):
	# Arguments:
		# data:					Point Cloud data (1 x num_points x 3)
		# shift:				Translation vector (1 x 3)

	try:
		data = np.asarray(data)
	except:
		pass
	return data+shift

# Apply the given transformation to given point cloud data.
def apply_transformation(datas,poses):			# Transformation function for (2 & 4c, loss 8b)
	# Arguments:
		# datas: 			Point Clouds (batch_size x num_points x 3)
		# poses: 			translation+euler (Batch_size x 6)
	# Output:
		# transformed_data: Transformed Point Clouds by given poses (batch_size x num_points x 3)
	transformed_data = np.copy(datas)
	for i in range(datas.shape[0]):
		transformed_data[i,:,:] = rotate_point_cloud_by_angle_z(transformed_data[i,:,:],poses[i,5])
		transformed_data[i,:,:] = rotate_point_cloud_by_angle_y(transformed_data[i,:,:],poses[i,4])
		transformed_data[i,:,:] = rotate_point_cloud_by_angle_x(transformed_data[i,:,:],poses[i,3])
		transformed_data[i,:,:] = translate(transformed_data[i,:,:],[poses[i,0],poses[i,1],poses[i,2]])
	return transformed_data

# Convert poses from 6D to 7D. 			# For loss function ( 8a )
def poses_euler2quat(poses):
	# Arguments:
		# poses: 			6D pose (translation + euler) (batch_size x 6)
	# Output: 
		# new_poses: 		7D pose (translation + quaternions) (batch_size x 7)

	new_poses = []					# Store 7D poses
	for i in range(poses.shape[0]):
		temp = t3d.euler2quat(poses[i,3],poses[i,4],poses[i,5])						# Convert from euler to quaternion. (1x4)
		temp1 = [poses[i,0],poses[i,1],poses[i,2],temp[0],temp[1],temp[2],temp[3]]		# Add translation & Quaternion (1x7)
		new_poses.append(temp1)												
	return np.asarray(new_poses)									

# Geenerate random poses equal to batch_size.
def generate_poses(batch_size):
	# Arguments:
		# batch_size:		No of 6D poses required.
	# Output:
		# poses:			Array of poses with translation and rotation (euler angles in radians) (batch_size x 6)

	poses = []					# List to store the 6D poses.
	for i in range(batch_size):
		# Generate random translations.
		x = np.round(2*np.random.random_sample()-1,2)
		y = np.round(2*np.random.random_sample()-1,2)
		z = np.round(2*np.random.random_sample()-1,2)
		# Generate random rotations.
		x_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2),3)
		y_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2),3)
		z_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2),3)
		poses.append([x,y,z,x_rot,y_rot,z_rot])
	return np.array(poses).reshape((batch_size,6))

# Convert 6D poses to transformation matrix.	# (for 4b)
def transformation(poses):		
	# Arguments:
		# poses: 					6D (x,y,z,euler_x,euler_y,euler_z) (in radians)
	# Output
		# transformation_matrix: 	batch_size x 4 x 4

	transformation_matrix = np.zeros((poses.shape[0],4,4))		
	transformation_matrix[:,3,3] = 1
	for i in range(poses.shape[0]):
		rot = t3d.euler2mat(poses[i,5],poses[i,4],poses[i,3],'szyx')	# Calculate rotation matrix using transforms3d
		transformation_matrix[i,0:3,0:3]=rot 							# Store rotation matrix in transformation matrix.
		transformation_matrix[i,0:3,3]=poses[i,0:3]						# Store translations in transformation matrix.
	return transformation_matrix

# Convert poses (quaternions) to transformation matrix and apply on point cloud.
def transformation_quat2mat(poses,TRANSFORMATIONS,templates_data):		# (for 4b)
	# Arguments:
		# poses: 					7D (x,y,z,quat_q0,quat_q1,quat_q2,quat_q3) (in radians) (batch_size x 7)
		# TRANSFORMATIONS: 			Overall tranformation matrix.
		# template_data: 			Point Cloud (batch_size x num_points x 3)
	# Output
		# TRANSFORMATIONS: 			Batch_size x 4 x 4
		# templates_data:			Transformed template data (batch_size x num_points x 3)
	
	poses = np.array(poses)												# Convert poses to array.
	poses = poses.reshape(poses.shape[-2],poses.shape[-1])
	for i in range(poses.shape[0]):
		transformation_matrix = np.zeros((4,4))
		transformation_matrix[3,3] = 1
		rot = t3d.quat2mat([poses[i,3],poses[i,4],poses[i,5],poses[i,6]])	# Calculate rotation matrix using transforms3d (library handles the normalization part of quaternion)
		transformation_matrix[0:3,0:3]=rot 									# Store rotation matrix in transformation matrix.
		transformation_matrix[0:3,3]=poses[i,0:3]							# Store translations in transformation matrix.
		TRANSFORMATIONS[i,:,:] = np.dot(transformation_matrix,TRANSFORMATIONS[i,:,:])		# 4b (Multiply tranfromation matrix to Initial Transfromation Matrix)
		templates_data[i,:,:]=np.dot(rot,templates_data[i,:,:].T).T 		# Apply Rotation to Template Data
		templates_data[i,:,:]=templates_data[i,:,:]+poses[i,0:3]			# Apply translation to template data
	return TRANSFORMATIONS,templates_data

# Convert the Final Transformation Matrix to Translation + Orientation (Euler Angles in Degrees)
def find_final_pose(TRANSFORMATIONS):
	# Arguments:
		# TRANSFORMATIONS: 			transformation matrix (batch_size x 4 x 4)
	# Output:
		# final_pose:				final pose predicted by network (batch_size x 6)

	final_pose = np.zeros((TRANSFORMATIONS.shape[0],6))		# Array to store the poses.
	for i in range(TRANSFORMATIONS.shape[0]):				
		rot = TRANSFORMATIONS[i,0:3,0:3]					# Extract rotation matrix.
		euler = t3d.mat2euler(rot,'szyx')					# Convert rotation matrix to euler angles. (Pre-multiplication)
		final_pose[i,3:6]=[euler[2],euler[1],euler[0]]		# Store the translation
		final_pose[i,0:3]=TRANSFORMATIONS[i,0:3,3].T 		# Store the euler angles.
	return final_pose

# Subtract the centroids from source and template (Like ICP) and then find the pose.
def centroid_subtraction(source_data, template_data):
	# Arguments:
		# source_data:			Source Point Clouds (batch_size x num_points x 3)
		# template_data:		Template Point Clouds (batch_size x num_points x 3)
	# Output:
		# source_data:					Centroid subtracted from source point cloud (batch_size x num_points x 3)
		# template_data:				Centroid subtracted from template point cloud (batch_size x num_points x 3)
		# centroid_translation_pose:	Apply this pose after final iteration. (batch_size x 7)

	centroid_translation_pose = np.zeros((source_data.shape[0],7))
	for i in range(source_data.shape[0]):
		source_centroid = np.mean(source_data[i],axis=0)
		template_centroid = np.mean(template_data[i],axis=0)
		source_data[i] = source_data[i] - source_centroid
		template_data[i] = template_data[i] - template_centroid
		centroid_translation = source_centroid - template_centroid
		centroid_translation_pose[i] = np.array([centroid_translation[0],centroid_translation[1],centroid_translation[2],1,0,0,0])
	return source_data, template_data, centroid_translation_pose


###################### Shuffling Operations #########################

# Randomly shuffle given array of poses for training procedure.
def shuffle_templates(templates):
	# Arguments:
		# templates:			Input array of templates to get randomly shuffled (batch_size x num_points x 3)
	# Output:
		# shuffled_templates:	Randomly ordered poses (batch_size x num_points x 3)

	shuffled_templates = np.zeros(templates.shape)						# Array to store shuffled templates.
	templates_idxs = np.arange(0,templates.shape[0])
	np.random.shuffle(templates_idxs)									# Randomly shuffle template indices.
	for i in range(templates.shape[0]):
		shuffled_templates[i,:,:]=templates[templates_idxs[i],:,:]		# Rearrange them as per shuffled indices.
	return shuffled_templates

# Randomly shuffle given array of poses for training procedure.
def shuffle_poses(poses):
	# Arguments:
		# poses:			Input array of poses to get randomly shuffled (batch_size x n)
	# Output:
		# shuffled_poses:	Randomly ordered poses (batch_size x n)

	shuffled_poses = np.zeros(poses.shape)				# Array to store shuffled poses.
	poses_idxs = np.arange(0,poses.shape[0])			
	np.random.shuffle(poses_idxs)						# Shuffle the indexes of poses.
	for i in range(poses.shape[0]):
		shuffled_poses[i,:]=poses[poses_idxs[i],:]		# Rearrange them as per shuffled indexes.
	return shuffled_poses

# Generate random transformation/pose for data augmentation.
def random_trans():
	# Output:
		# 6D pose with first 3 translation values and last 3 euler angles in radian about x,y,z-axes. (1x6)

	# Generate random translations.
	x_trans, y_trans, z_trans = 0.4*np.random.uniform()-0.2, 0.4*np.random.uniform()-0.2, 0.4*np.random.uniform()-0.2	
	# Generate random rotation angles.
	x_rot, y_rot, z_rot = (np.pi/9)*np.random.uniform()-(np.pi/18), (np.pi/9)*np.random.uniform()-(np.pi/18), (np.pi/9)*np.random.uniform()-(np.pi/18)
	return [x_trans,y_trans,z_trans,x_rot,y_rot,z_rot]

# Generate random poses for each batch to train the network.
def generate_random_poses(batch_size):
	# Arguments:
		# Batch_size:		No of poses in the output
	# Output:
		# poses:			Randomly generated poses (batch_size x 6)

	poses = []
	for i in range(batch_size):
		x_trans, y_trans, z_trans = 2*np.random.uniform()-1, 2*np.random.uniform()-1, 2*np.random.uniform()-1										# Generate random translation
		x_rot, y_rot, z_rot = (np.pi)*np.random.uniform()-(np.pi/2), (np.pi)*np.random.uniform()-(np.pi/2), (np.pi)*np.random.uniform()-(np.pi/2)	# Generate random orientation
		poses.append([np.round(x_trans,4), np.round(y_trans,4), np.round(z_trans,4), np.round(x_rot,4), np.round(y_rot,4), np.round(z_rot,4)])		# round upto 4 decimal digits
	return np.array(poses)

###################### Tensor Operations #########################

def rotate_point_cloud_by_angle_y_tensor(data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
		Input:
		  Nx3 array, original batch of point clouds
		Return:
		  Nx3 array, rotated batch of point clouds
	"""
	cosval = tf.cos(rotation_angle)
	sinval = tf.sin(rotation_angle)
	rotation_matrix = tf.reshape([[cosval, 0, sinval],[0, 1, 0],[-sinval, 0, cosval]], [3,3])
	data = tf.reshape(data, [-1, 3])
	rotated_data = tf.transpose(tf.tensordot(rotation_matrix, tf.transpose(data), [1,0]))
	return rotated_data

def rotate_point_cloud_by_angle_x_tensor(data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
		Input:
		  Nx3 array, original batch of point clouds
		Return:
		  Nx3 array, rotated batch of point clouds
	"""
	cosval = tf.cos(rotation_angle)
	sinval = tf.sin(rotation_angle)
	rotation_matrix = tf.reshape([[1, 0, 0],[0, cosval, -sinval],[0, sinval, cosval]], [3,3])
	data = tf.reshape(data, [-1, 3])
	rotated_data = tf.transpose(tf.tensordot(rotation_matrix, tf.transpose(data), [1,0]))
	return rotated_data

def rotate_point_cloud_by_angle_z_tensor(data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
		Input:
		  Nx3 array, original batch of point clouds
		Return:
		  Nx3 array, rotated batch of point clouds
	"""
	cosval = tf.cos(rotation_angle)
	sinval = tf.sin(rotation_angle)
	rotation_matrix = tf.reshape([[cosval, -sinval, 0],[sinval, cosval, 0],[0, 0, 1]], [3,3])
	data = tf.reshape(data, [-1, 3])
	rotated_data = tf.transpose(tf.tensordot(rotation_matrix, tf.transpose(data), [1,0]))
	return rotated_data

def translate_tensor(data,shift):
	# Add the translation vector to given tensor. (num_point x 3)
	return tf.add(data,shift)	

# Tranform the data as per given poses with orientation as euler in degrees.
def transformation_tensor(datas,poses):
	# Arguments:
		# datas: 				Tensor of Point Cloud (batch_size x num_points x 3)
		# poses: 				Tensor of Poses (translation + euler angles in degrees) (batch_size x num_points x 3)
	# Ouput:
		# transformed_data:		Tensor of transformed point cloud (batch_size x num_points x 3)

	transformed_data = tf.zeros([datas.shape[1], datas.shape[2]])		# Tensor to store the transformed point clouds as tensor.
	for i in range(datas.shape[0]):
		transformed_data_t = rotate_point_cloud_by_angle_x_tensor(datas[i,...],poses[i,3])				# Rotate about x-axis
		transformed_data_t = rotate_point_cloud_by_angle_y_tensor(transformed_data_t,poses[i,4])		# Rotate about y-axis
		transformed_data_t = rotate_point_cloud_by_angle_z_tensor(transformed_data_t,poses[i,5])		# Rotate about z-axis
		transformed_data_t = translate_tensor(transformed_data_t,[poses[i,0],poses[i,1],poses[i,2]])	# Translate by given vector.
		transformed_data = tf.concat([transformed_data, transformed_data_t], 0)							# Append the transformed tensor point cloud.
	transformed_data = tf.reshape(transformed_data, [-1, datas.shape[1], datas.shape[2]])[1:]			# Reshape the transformed tensor and remove first one. (batch_size x num_point x 3)
	return transformed_data

# Tranform the data as per given poses with orientation as quaternion.
def transformation_quat_tensor(data,quat,translation):
	# Arguments:
		# data:					Tensor of Point Cloud. (batch_size x num_point x 3)
		# quat:					Quaternion tensor to generate rotation matrix.	(batch_size x 4)
		# translation:			Translation tensor to translate the point cloud. (batch_size x 3)
	# Outputs:
		# transformed_data: 	Tensor of Rotated and Translated Point Cloud Data. (batch_size x num_points x 3)

	transformed_data = tf.zeros([data.shape[1],3])		# Tensor to store transformed data.
	for i in range(quat.shape[0]):
		# Seperate each quaternion value.
		q0 = tf.slice(quat,[i,0],[1,1])
		q1 = tf.slice(quat,[i,1],[1,1])
		q2 = tf.slice(quat,[i,2],[1,1])
		q3 = tf.slice(quat,[i,3],[1,1])
		# Convert quaternion to rotation matrix.
		# Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
			  # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
		R = [[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
			 [2*(q1*q2+q0*q3), q0*q0+q2*q2-q1*q1-q3*q3, 2*(q2*q3-q0*q1)],
			 [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0+q3*q3-q1*q1-q2*q2]]

		R = tf.reshape(R,[3,3]) 			# Convert R into a single tensor of shape 3x3.
		# tf.tensordot: Arg: tensor1, tensor2, axes
		# axes defined for tensor1 & tensor2 should be of same size.
		# axis 1 of R is of size 3 and axis 0 of data (3xnum_points) is of size 3.

		temp_rotated_data = tf.transpose(tf.tensordot(R, tf.transpose(data[i,...]), [1,0]))		# Rotate the data. (num_points x 3)
		temp_rotated_data = tf.add(temp_rotated_data,translation[i,...])						# Add the translation (num_points x 3)
		transformed_data = tf.concat([transformed_data, temp_rotated_data],0)					# Append data (batch_size x num_points x 3)
	transformed_data = tf.reshape(transformed_data, [-1,data.shape[1],3])[1:]					# Reshape data and remove first point cloud. (batch_size x num_point x 3)
	return transformed_data



###################### Display Operations #########################

# Display data inside ModelNet files.
def display_clouds(filename,model_no):
	# Arguments:
		# filename:			Name of file to read the data from. (string)
		# model_no:			Number to choose the model inside that file. (int)

	data = []
	# Read the entire data from that file.
	with open(os.path.join('data','templates',filename),'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			row = [float(x) for x in row]
			data.append(row)
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	data = np.asarray(data)

	start_idx = model_no*2048
	end_idx = (model_no+1)*2048
	data = data[start_idx:end_idx,:]		# Choose specific data related to the given model number.

	X,Y,Z = [],[],[]
	for row in data:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	ax.scatter(X,Y,Z)
	plt.show()

# Display given Point Cloud Data in blue color (default).
def display_clouds_data(data):
	# Arguments:
		# data: 		array of point clouds (num_points x 3)

	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	try:
		data = data.tolist()
	except:
		pass
	X,Y,Z = [],[],[]
	for row in data:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	ax.scatter(X,Y,Z)
	plt.show()

# Display given template, source and predicted point cloud data.
def display_three_clouds(data1,data2,data3,title):
	# Arguments:
		# data1 		Template Data (num_points x 3) (Red)
		# data2			Source Data (num_points x 3) (Green)
		# data3			Predicted Data (num_points x 3) (Blue)

	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	try:
		data1 = data1.tolist()
		data2 = data2.tolist()
		data3 = data3.tolist()
	except:
		pass
	# Add Template Data in Plot
	X,Y,Z = [],[],[]
	for row in data1:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l1 = ax.scatter(X,Y,Z,c=[1,0,0,1])
	# Add Source Data in Plot
	X,Y,Z = [],[],[]
	for row in data2:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l2 = ax.scatter(X,Y,Z,c=[0,1,0,1])
	# Add Predicted Data in Plot
	X,Y,Z = [],[],[]
	for row in data3:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l3 = ax.scatter(X,Y,Z,c=[0,0,1,1])

	# Add details to Plot.
	plt.legend((l1,l2,l3),('Template Data','Source Data','Predicted Data'),prop={'size':15},markerscale=4)
	ax.tick_params(labelsize=10)
	ax.set_xlabel('X-axis',fontsize=15)
	ax.set_ylabel('Y-axis',fontsize=15)
	ax.set_zlabel('Z-axis',fontsize=15)
	plt.title(title,fontdict={'fontsize':25})
	ax.xaxis.set_tick_params(labelsize=15)
	ax.yaxis.set_tick_params(labelsize=15)
	ax.zaxis.set_tick_params(labelsize=15)
	plt.show()

# Display given template, source and predicted point cloud data.
def display_two_clouds(data1,data2):
	# Arguments:
		# data1 		Template Data (num_points x 3) (Red)
		# data2			Source Data (num_points x 3) (Green)
		# data3			Predicted Data (num_points x 3) (Blue)

	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	try:
		data1 = data1.tolist()
		data2 = data2.tolist()
		data3 = data3.tolist()
	except:
		pass
	# Add Template Data in Plot
	X,Y,Z = [],[],[]
	for row in data1:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l1 = ax.scatter(X,Y,Z,c=[1,0,0,1])
	# Add Source Data in Plot
	X,Y,Z = [],[],[]
	for row in data2:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l2 = ax.scatter(X,Y,Z,c=[0,0,1,1])

	# Add details to Plot.
	plt.legend((l1,l2),('Input Data','Output Data'),prop={'size':15},markerscale=4)
	ax.tick_params(labelsize=10)
	ax.set_xlabel('X-axis',fontsize=15)
	ax.set_ylabel('Y-axis',fontsize=15)
	ax.set_zlabel('Z-axis',fontsize=15)
	# plt.title(title,fontdict={'fontsize':25})
	ax.xaxis.set_tick_params(labelsize=15)
	ax.yaxis.set_tick_params(labelsize=15)
	ax.zaxis.set_tick_params(labelsize=15)
	ax.set_axis_off()
	plt.show()

# Display template, source, predicted point cloud data with results after each iteration.
def display_itr_clouds(data1,data2,data3,ITR,title):
	# Arguments:
		# data1 		Template Data (num_points x 3) (Red)
		# data2			Source Data (num_points x 3) (Green)
		# data3			Predicted Data (num_points x 3) (Blue)
		# ITR 			Point Clouds obtained after each iteration (iterations x batch_size x num of points x 3) (Yellow)

	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	print(ITR.shape)		# Display Number of Point Clouds in ITR.
	try:
		data1 = data1.tolist()
		data2 = data2.tolist()
		data3 = data3.tolist()
	except:
		pass
	# Add Template Data in Plot
	X,Y,Z = [],[],[]
	for row in data1:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l1 = ax.scatter(X,Y,Z,c=[1,0,0,1])
	# Add Source Data in Plot
	X,Y,Z = [],[],[]
	for row in data2:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l2 = ax.scatter(X,Y,Z,c=[0,1,0,1])
	# Add Predicted Data in Plot
	X,Y,Z = [],[],[]
	for row in data3:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l3 = ax.scatter(X,Y,Z,c=[0,0,1,1])
	# Add point clouds after each iteration in Plot.
	for itr_data in ITR:
		X,Y,Z = [],[],[]
		for row in itr_data[0]:
			X.append(row[0])
			Y.append(row[1])
			Z.append(row[2])
		ax.scatter(X,Y,Z,c=[1,1,0,0.5])

	# Add details to Plot.
	plt.legend((l1,l2,l3),('Template Data','Source Data','Predicted Data'),prop={'size':15},markerscale=4)
	ax.tick_params(labelsize=10)
	ax.set_xlabel('X-axis',fontsize=15)
	ax.set_ylabel('Y-axis',fontsize=15)
	ax.set_zlabel('Z-axis',fontsize=15)
	plt.title(title,fontdict={'fontsize':25})
	ax.xaxis.set_tick_params(labelsize=15)
	ax.yaxis.set_tick_params(labelsize=15)
	ax.zaxis.set_tick_params(labelsize=15)
	plt.show()





if __name__=='__main__':
	# a = np.array([[0,0,0,0,0,0],[0,0,0,90,0,0]])
	# print a.shape
	# a = poses_euler2quat(a)
	# print(a[1,3]*a[1,3]+a[1,4]*a[1,4]+a[1,5]*a[1,5]+a[1,6]*a[1,6])
	# print(a[0,3]*a[0,3]+a[0,4]*a[0,4]+a[0,5]*a[0,5]+a[0,6]*a[0,6])
	# print a.shape
	# display_clouds('airplane_templates.csv',0)

	templates = helper.process_templates('multi_model_templates')
	# templates = helper.process_templates('templates')
	# airplane = templates[0,:,:]
	idx = 199
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	# start = idx*2048
	# end = (idx+1)*2048
	ax.scatter(templates[idx,:,0],templates[idx,:,1],templates[idx,:,2])
	plt.show()
	print(templates.shape)

