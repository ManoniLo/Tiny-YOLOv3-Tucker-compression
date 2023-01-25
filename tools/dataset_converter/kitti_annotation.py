import os, argparse
import numpy as np
import random
import sys

sys.path.append('../../')

from common.utils import get_dataset

seed_value = 1
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONASHSEED'] = str(seed_value)

ann_path = '../../annotations'
trainval_ann_file = 'kitti_trainval.txt'
train_ann_file = 'kitti_train.txt'
val_ann_file = 'kitti_val.txt'


sets = ['train', 'val']

split = [0.9, 0.1]
dataset = get_dataset(os.path.join(ann_path,trainval_ann_file), shuffle = True, seed = seed_value)
num_data = len(dataset)
num_train, num_val = round(split[0]*num_data), round(split[1]*num_data)

split_datasets = dataset[:num_train], dataset[num_train:]




# files with train/val image names..........................
for i,dset in enumerate(sets):
    ann_name = 'kitti_' + dset + '.txt'
    ann_file = os.path.join(ann_path,ann_name)
    with open(ann_file,'w') as f:
        for line in split_datasets[i]:
            f.writelines(line+'\n')
    














