import os, argparse
import numpy as np
import random

trainval_ann_file = '../../annotations/kitti_trainval.txt'
ann_info_path = '../../annotations/kitti_split'
ann_path = '../../annotations'
sets = ['train', 'val']

ann_sep = " "
sep = '/'
image_dir = 'data_object_image_2/training/image_2'

trainval_data = []
trainval_names = []


with open(trainval_ann_file,'r') as f:
    for line in f:
        line = line.split()
        trainval_data.append(line[1:])
        trainval_names.append(line[0].split('/')[-1])


trainval_names = np.array(trainval_names)
trainval_data = np.array(trainval_data)

# files with train/val image names..........................
for dset in sets:
    info_name = 'kitti_' + dset + '_names.txt'
    ann_name = 'kitti_' + dset + '.txt'
    info_file = os.path.join(ann_info_path,info_name)
    ann_file = os.path.join(ann_path,ann_name)


    with open(info_file,'r') as f:
        fnames = np.array([line.split()[0] + ".png" for line in f])



    # select train/val lines from trainval.txt.................
    ann_mask = np.array([name in fnames for name in trainval_names])
    ann_names = trainval_names[ann_mask]
    ann_data = trainval_data[ann_mask]
    
    ann_names = np.array([sep.join([image_dir,name]) for name in ann_names])
    ann_lines = [ann_sep.join([ann_names[i]] + ann_data[i] + ["\n"]) for i in range(len(ann_names))]

    ann_lines = np.array(ann_lines)


    
    with open(ann_file,'w') as f:
        f.writelines(ann_lines)
    

















