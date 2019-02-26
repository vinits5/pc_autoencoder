# Auto Encoder for 3D Point Clouds

### Network Structure:
<p align="center">
	<img src="https://github.com/vinits5/pc_autoencoder/blob/master/results/network_structure.png">
</p>

### Code:
Steps to train the auto-encoder:
1. Download ModelNet40 Dataset [[Link]](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)
2. Clone repository.
3. Extract the zip file and copy *modelnet40_ply_hdf5_2048* folder to *pc_autoencoder/data*.
4. *python pointnet_autoencoder_train.py --mode train*

Steps to test the auto-encoder:
1. Download dataset as given in training steps.
2. Download weights for the trained network. [[Link]](https://drive.google.com/drive/folders/17k0mWR65eHQbnWcvNWJKlZ1hXeVYVhm7?usp=sharing)
3. *python pointnet_autoencoder_train.py --mode test*

Visualise the Dataset:
*python show_pc.py idx*
idx: Index of Point Cloud in ModelNet40 Dataset.

### Results:
**Red colored point clouds are input to the network and blue point clouds are the output.**

[Note: A translation has been applied to blue point clouds during testing for a better visualisation purpose.]
<p align="center">
	<img src="https://github.com/vinits5/pc_autoencoder/blob/master/results/result1.png">
</p>

#### Additional Results:

<p align="center">
	<img src="https://github.com/vinits5/pc_autoencoder/blob/master/results/result2.png">
	<img src="https://github.com/vinits5/pc_autoencoder/blob/master/results/result3.png">
	<img src="https://github.com/vinits5/pc_autoencoder/blob/master/results/result4.png">
</p>

### References:
1. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [Link](https://arxiv.org/abs/1612.00593)
2. PCN: Point Completion Network [[Link]](https://arxiv.org/abs/1808.00671)