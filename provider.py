import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
	os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
	www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
	zipfile = os.path.basename(www)
	os.system('wget %s; unzip %s' % (www, zipfile))
	os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
	os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
	""" Shuffle data and labels.
		Input:
		  data: B,N,... numpy array
		  label: B,... numpy array
		Return:
		  shuffled data, label and shuffle indices
	"""
	idx = np.arange(len(labels))
	np.random.shuffle(idx)
	return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
	""" Randomly rotate the point clouds to augument the dataset
		rotation is per shape based along up direction
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]])
		shape_pc = batch_data[k, ...]
		rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
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
		rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data

def rotate_point_cloud_by_angle_x_axis(batch_data, rotation_angle):
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
		rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data

def rotate_point_cloud_by_angle_z_axis(batch_data, rotation_angle):
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
		rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
	""" Randomly jitter points. jittering is per point.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, jittered batch of point clouds
	"""
	B, N, C = batch_data.shape
	assert(clip > 0)
	jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
	jittered_data += batch_data
	return jittered_data

def getDataFiles(list_filename):
	return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
	f = h5py.File(h5_filename)
	data = f['data'][:]
	label = f['label'][:]
	return (data, label)

def loadDataFile(filename):
	return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
	f = h5py.File(h5_filename)
	data = f['data'][:]
	label = f['label'][:]
	seg = f['pid'][:]
	return (data, label, seg)


def loadDataFile_with_seg(filename):
	return load_h5_data_label_seg(filename)


# Defined by Vinit
def zero_mean(data):
	centroid = sum(data)/(data.shape[0]*1.0)
	return data-centroid

def normalize(data):
	x,y,z = data[:,0],data[:,1],data[:,2]
	x_max,y_max,z_max,x_min,y_min,z_min = max(x),max(y),max(z),min(x),min(y),min(z)
	x = x-x_min
	y = y-y_min
	z = z-z_min
	x = x/((x_max-x_min)*1.0)
	y = y/((y_max-y_min)*1.0)
	z = z/((z_max-z_min)*1.0)
	data = np.zeros((1024,3))
	print max(x),max(y),max(z),min(x),min(y),min(z)
	data[:,0],data[:,1],data[:,2]=x,y,z
	return data

def scale(data,scale_x,scale_y,scale_z):
	try:
		data = np.asarray(data)
	except:
		pass
	scale = [scale_x,scale_y,scale_z]
	return data*scale

def translate(data,shift):
	try:
		data = np.asarray(data)
	except:
		pass
	# shift = [shift_x,shift_y,shift_z]
	return data+shift

def partial_data_plane(airplane_data):
	airplane_data_f1,airplane_data_f2 = [],[]
	for i in range(airplane_data.shape[0]):
		if airplane_data[i,2]>-0.0363:
			airplane_data_f1.append(airplane_data[i,:])
		else:
			airplane_data_f2.append(airplane_data[i,:])
	return np.asarray(airplane_data_f1),np.asarray(airplane_data_f2)

def clone_cloud(data):
	clone_data = np.zeros((32,2048,3))
	for i in range(32):
		clone_data[i,:,:]=current_data[1,:,:]
	return clone_data

import numpy.linalg as alg
def find_lambda(f1,f2,f):
	f1 = np.matrix(f1.reshape((1024,1)))
	f2 = np.matrix(f2.reshape((1024,1)))
	f = np.matrix(f.reshape((1024,1)))
	B = f-f2
	A = f1-f2
	Atrans = A.getT()
	term1 = alg.inv(Atrans*A)
	term2 = Atrans*B
	return float(term1*term2)

def display_clouds_data(data):
	import matplotlib.pyplot as plt 
	from mpl_toolkits.mplot3d import Axes3D
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