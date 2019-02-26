import provider
import helper

TRAIN_FILES = provider.getDataFiles('data/modelnet40_ply_hdf5_2048/train_files.txt')

current_data, current_label = provider.loadDataFile(TRAIN_FILES[0])

idx = int(sys.argv[1])

helper.display_clouds_data(current_data[idx])