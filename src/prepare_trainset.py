
# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# Revised by Reza Azad

import sys
sys.path.append(r'/home/nathanmolinier/data_nvme/code/spinalcordtoolbox')
from Data2array import load_Data_Bids2Array, load_Data_Bids2Array_with_subjects
from train_utils import extract_groundtruth_heatmap, extract_groundtruth_heatmap_with_subjects_and_GT_coords
#from models import *
import numpy as np
import copy
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch arguments')
parser.add_argument('--source', type=str,
                        help='Pth to Dataset')                     
parser.add_argument('--target', type=str,
                        help='Pth to save the prepared dataset')  
parser.add_argument('--mode', default=2, type=int,
                        help='#0 for both t1 and t2 , 1 for t1 only , 2 for t2 only')
                        
args = parser.parse_args()                        
                        
print('load dataset: ',os.path.abspath(args.source))
ds_train = load_Data_Bids2Array(args.source, mode= args.mode, split='train', aim='full')
ds_test = load_Data_Bids2Array_with_subjects(args.source, mode= args.mode, split='test', aim='full')  # we want to keep track of the subjects name
print('creating heatmap')
full_train = extract_groundtruth_heatmap(ds_train)
full_test = extract_groundtruth_heatmap_with_subjects_and_GT_coords(ds_test)  # we want to keep track of the subjects name and the ground truth position of the vertebral discs
print('saving the prepared dataset')
if args.mode == 0:
     modality = 't1_t2'
if args.mode == 1:
     modality = 't1'
if args.mode == 2:
     modality = 't2'

output_trainset = args.target + '_train_' + modality
with open(output_trainset, 'wb') as file_pi:
     pickle.dump(full_train, file_pi)
     
output_testset = args.target + '_test_' + modality
with open(output_testset, 'wb') as file_pi:
     pickle.dump(full_test, file_pi)

print('finished')   

