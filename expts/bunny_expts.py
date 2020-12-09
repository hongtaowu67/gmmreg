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
def load_groundtruth_partial(groundtruth_csv, data_name_list, num_pair=30):
    """The groundtruth is saved in a csv.
        It is in the form of 4k * (4*30) where k is the data number.
        Args:
        - groundtruth_csv: path to the groundtruth csv
        - data_name_list: list of data prefixs
        - num_pair: number of pair for each experiments
        Returns:
        - gt_mat: dictionary of numpy matrix containing
                  the groundtruth. Key is the data_prefix,
                  value is the numpy matrix. Each matrix is
                  4 x (4*30)
    """
    gt_map = {}
    with open(groundtruth_csv, 'r') as f:
        reader = csv.reader(f)
        lines = [row for row in reader]
        for data_group_idx, data_name in enumerate(data_name_list):
            gt_mat = np.zeros((4, 4*num_pair))
            for i in range(4):
                row = lines[data_group_idx * 4 + i]
                assert len(row) == 4 * num_pair
                for col_idx, item in enumerate(row):
                    gt_mat[i % 4, col_idx] = float(item)    
            gt_map[data_name] = gt_mat              

    return gt_map

# https://github.com/hongtaowu67/LG-GMM/blob/LGGMM/benchmark/benchmark_utils.py
def matrix_frobenius_norm(m1, m2):
    """Calculate frobenius norm between two matrices.
        Args: 
        - m1: numpy array
        - m2: numpy array
    """
    assert m1.shape == m2.shape
    return np.sum((m1-m2) * (m1-m2))

def run_rigid_batch(data_path, data_prefix_1, data_prefix_2, config_path, num_pair=30):
    matrix = []
    matrix_inv = []
    model_txt = os.path.join(data_path, data_prefix_1 + "_source.txt")
    for i in range(num_pair):
        scene_txt = os.path.join(data_path, data_prefix_2 + "_target" + str(i+1) + ".txt")
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
    DATA_PATH = '../data/bunny/txt'
    CONFIG_FILE = './dragon_stand.ini'
    GT_PATH = '../data/bunny/gt_rand.csv'
    data_list = ['bun000',   # 0
                 'bun045',   # 1
                 'bun090',   # 2
                 'bun180',   # 3
                 'bun270',   # 4
                 'bun315',   # 5
                 'chin',     # 6
                 'ear_back', # 7
                 'top2',     # 8
                 'top3']     # 9
    num_pair = 30
    success_threshold = 0.3

    data_pair = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                 (0, 6), (6, 5), (0, 9), (1, 9), (2, 9), (2, 7),
                 (8, 7), (8, 3), (8, 2), (8, 9), (3, 7)]
    
    result = {}
    result_txt = '../data/bunny/result_1101.txt'

    for pair in data_pair:
        target_idx = pair[0]
        source_idx = pair[1]

        gt_map = load_groundtruth_partial(GT_PATH, data_list)
        reg_mat, reg_mat_inv = run_rigid_batch(DATA_PATH, data_list[target_idx], 
            data_list[source_idx], CONFIG_FILE, num_pair=num_pair)
        gt_mat = gt_map[data_list[source_idx]]
        
        total_error = 0
        success = 0
        for i in range(num_pair):
            reg_mat_i = reg_mat[i][:3, :3]
            gt_mat_i = gt_mat[:3, (4*i):(4*i+3)]
            error = matrix_frobenius_norm(reg_mat_i, gt_mat_i)
            if error < success_threshold:
                success += 1
                total_error += error
        success_rate = success / num_pair
        average_err  = total_error / success
        result[(target_idx, source_idx)] = (success_rate, average_err)
        print("Success rate: {}/{}".format(success, num_pair))
        print("Average error: {}".format(total_error / success))
    
    for pair in result:
        succ_rate, avg_err = result[pair]
        print("{}->{}: {}, {}".format(data_list[pair[0]], data_list[pair[1]], succ_rate, avg_err))

    
        

