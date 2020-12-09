#!/usr/bin/env python
#coding=utf-8

"""
This Python script can be used to test the gmmreg algorithm on the Stanford
"dragon stand" dataset as described in Section 6.1 of the gmmreg PAMI paper.

Benchmarking on the partially overlapping data
Modified by Hongtao Wu on Nov 1, 202
"""

import os, subprocess, time, csv
import numpy as np

from common_utils import *

# https://github.com/bistromath/gr-air-modes/blob/master/python/Quaternion.py
from Quaternion import Quat, normalize

# https://github.com/hongtaowu67/LG-GMM/blob/LGGMM/benchmark/benchmark_utils.py
def load_groundtruth_outlier(groundtruth_txt, num_pair=30):
    """ The ground truth is saved in a txt.
        The txt is a 4 x 4n matrix.
        
        Args:
        - groundtruth_txt: path to the groundtruth txt
        - num_pair: number of pair in the experiments
        Returns:
        - gt_mat: numpy matrix containing the groundtruth
    """
    
    with open(groundtruth_txt, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 4

    line_str = []
    for line in lines:
        line = line.split(',')
        line_str.append(line)
    
    # number of benchmark
    assert len(line_str[0]) / 4 == num_pair
    
    gt_mat = np.zeros((4, 4 * num_pair))
    for i in range(num_pair):
        for j in range(4):
            gt_mat[j, 4*i]   = float(line_str[j][4*i])
            gt_mat[j, 4*i+1] = float(line_str[j][4*i+1])
            gt_mat[j, 4*i+2] = float(line_str[j][4*i+2])
            gt_mat[j, 4*i+3] = float(line_str[j][4*i+3])
    
    return gt_mat

# https://github.com/hongtaowu67/LG-GMM/blob/LGGMM/benchmark/benchmark_utils.py
def matrix_frobenius_norm(m1, m2):
    """Calculate frobenius norm between two matrices.
        Args: 
        - m1: numpy array
        - m2: numpy array
    """
    assert m1.shape == m2.shape
    return np.sum((m1-m2) * (m1-m2))

def run_rigid_batch(data_path, group_name, config_path, num_pair=30):
    matrix = []
    matrix_inv = []
    for i in range(num_pair):
        model_txt = "GOutRatio" + "_" + group + "_" + str(i+1) + "_Base.txt" 
        scene_txt = "GOutRatio" + "_" + group + "_" + str(i+1) + "_Rand.txt" 
        model_txt = os.path.join(data_path, model_txt)
        scene_txt = os.path.join(data_path, scene_txt)
        _, matrix1, _, _ = run_rigid_pairwise(BINARY_FULLPATH, model_txt, scene_txt, config_path)
        _, matrix2, _, _ = run_rigid_pairwise(BINARY_FULLPATH, scene_txt, model_txt, config_path)

        matrix.append(matrix1)
        matrix_inv.append(matrix2)
    return matrix, matrix_inv


# Run pair-wise registrations, record errors and run time.
def main():
    raise NotImplemented



import sys
if __name__ == "__main__":
    DATA_PATH = '../data/outlier_data/Gaussian_txt'
    CONFIG_FILE = './dragon_stand.ini'
    GT_PATH = '../data/outlier_data/Gaussian_ply/Goutlier_gt.csv'
    data_list = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', 
                 '0.6', '0.7', '0.8', '0.9', '1']
    num_pair = 30
    success_threshold = 0.3

    gt_mat = load_groundtruth_outlier(GT_PATH)
    
    result = {}

    for group_idx, group in enumerate(data_list):
        total_error = 0
        success = 0
        reg_mat, reg_mat_inv = run_rigid_batch(DATA_PATH, group, CONFIG_FILE, num_pair=num_pair)
        
        for i in range(num_pair):
            reg_mat_i = reg_mat[i][:3, :3]
            gt_mat_i = gt_mat[:3, (4*i):(4*i+3)]
            print(reg_mat_i)
            print(gt_mat_i)
            error = matrix_frobenius_norm(reg_mat_i, gt_mat_i)
            if error < success_threshold:
                success += 1
                total_error += error
            print ("=====")

        success_rate = success / num_pair
        average_err  = total_error / success
        result[group] = (success_rate, average_err)
        print("Success rate: {}/{}".format(success, num_pair))
        print("Average error: {}".format(total_error / success))
    
    for group in result:
        succ_rate, avg_err = result[group]
        print("Gaussian Outlier {}: {}, {}".format(group, succ_rate, avg_err))

    
        

