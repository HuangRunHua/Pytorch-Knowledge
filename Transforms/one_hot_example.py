import numpy as np
"""
本程序展示标签生成one-hot向量的过程
6个数据集所属类别标签如下:
    - [0,3,2,3,1,1]
数据集的种类为4种(0~3)，因此生成的矩阵为6x4，每一行只有一个1，结果如下:
    - [ [1. 0. 0. 0.]   标签为0的数据集
        [0. 0. 0. 1.]   标签为3的数据集
        [0. 0. 1. 0.]   标签为2的数据集
        [0. 0. 0. 1.]   标签为3的数据集
        [0. 1. 0. 0.]   标签为1的数据集
        [0. 1. 0. 0.]]  标签为1的数据集
"""
# 假设6个数据集如下，其中[0,3,2,3,1,1]分别表示数据集所属于的类别
labels_dense = np.asarray([0,3,2,3,1,1]) 
# 假设数据集的种类为4种
num_classes = 4
num_labels = labels_dense.shape[0]
index_offset = np.arange (num_labels) * num_classes
labels_one_hot = np.zeros((num_labels, num_classes))
labels_one_hot.flat[index_offset + labels_dense.ravel()]= 1
# 将桥签对应的向量位置置1(拉平以后对应1的位置为0,7,10,15,17,21）
# flat0:1-D iterator over the array, raval:拉平向量
print(labels_one_hot)