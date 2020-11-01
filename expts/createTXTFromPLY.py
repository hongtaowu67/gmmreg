"""
Create txt file of the point cloud.
For testing the GMMReg.

Written by Hongtao Wu on Nov 1, 2020
"""
import os
import open3d as o3
import numpy as np

def main(ply_path, txt_dir):
    pc = o3.io.read_point_cloud(ply_path)
    pc_np_array = np.asarray(pc.points)

    ply_file = ply_path.split('/')[-1]
    pc_name = ply_file.split('.')[0]
    txt_file = pc_name + ".txt"
    txt_path = os.path.join(txt_dir, txt_file)

    np.savetxt(txt_path, pc_np_array)
    print("Finish processing {}".format(pc_name))

if __name__ == "__main__":
    ply_dir = "../data/bunny/ply"
    txt_dir = "../data/bunny/txt"
    file_list = os.listdir(ply_dir)

    for f in file_list:
        if '.ply' in f:
            ply_path = os.path.join(ply_dir, f)
            main(ply_path, txt_dir)
    