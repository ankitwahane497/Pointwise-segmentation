import sys
sys.path.append('/home/sanket/MS_Thesis/Pointwise-segmentation/models/tf_utils')
import edge_tf_util as tf_util
# from sklearn.neighbors import KDTree
import pdb
import numpy as np
from scipy.spatial import cKDTree
# from pyflann import *

def global_nearest_neighbors_features(data, k = 20):
    tree = [ KDTree(data[i], leaf_size=2) for i in range(len(data))]
    data_ = [tree[i].query(data[i], k= k) for i in range(len(data))]
    nn_pts = [data[i][data_[i][1]] for i in range(len(data))]
    nn_pts =  np.array(nn_pts)
    return nn_pts


def get_global_features(data,knn):
    nn_pts = [np.squeeze(data[i][knn[i]],axis = -2) for i in range(len(data))]
    nn_pts =  np.array(nn_pts)
    if nn_pts.shape[-1] == 1:
        nn_pts = np.squeeze(nn_pts , axis  = -1)
    #pdb.set_trace()
    return nn_pts

def get_global_features_deep(data, knn):
    nn_pts = [np.mean(data[i][knn],axis = -2) for i in range(len(data))]
    nn_pts = np.array(nn_pts)
    return nn_pts

def get_local_features(data,knn):
    #pdb.set_trace()
    nn_pts = [np.repeat(data[i],10,axis =-2) - np.squeeze(data[i][knn[i]],axis =-2)  for i in range(len(data))]
    #nn_pts = [data[i] - data[i][knn]  for i in range(len(data))]
    nn_pts =  np.array(nn_pts)
    if nn_pts.shape[-1] == 1:
        nn_pts = np.squeeze(nn_pts, axis  = -1)
    #pdb.set_trace()
    return nn_pts

def get_local_features_deep(data,knn):
    #     nn_pts  = np.zeros_like(data)
    #     nn_pts = data[:][:,np.newaxis] - data[:][knn]
    #pdb.set_trace()
    nn_pts = [data[i] - np.mean(data[i][knn],axis = -2)  for i in range(len(data))]
    nn_pts =  np.array(nn_pts)
    return nn_pts


def get_nearest_neighbors_id(data, k =20):
    data = np.squeeze(data, axis = -2)
    # tree = [ KDTree(data[i], leaf_size=2) for i in range(len(data))]
    tree = [ cKDTree(data[i]) for i in range(len(data))]
    data_ = [tree[i].query(data[i], k = k) for i in range(len(data))]
    data_ = np.array(data_).astype(np.int32)
    return data_[:,1]

def local_nearest_neighbors_features(data, k = 20):
    tree   = [KDTree(data[i], leaf_size=100) for i in range(len(data))]
    data_  = [tree[i].query(data[i], k= 30) for i in range(len(data))]
    nn_pts = [data[i][:,np.newaxis] - data[i][data_[i][1]]  for i in range(len(data))]
    # nn_pts = [data[:][:,np.newaxis] - data[:][data_[i][1]]]
    nn_pts =  np.array(nn_pts)
    return nn_pts


def feature_network(data, is_training, bn_decay , mlp = [32,32], name = 'Layer_'):
    layers = []
    for i in range(len(mlp)):
        if len(layers) == 0:
            tmp_layer = tf_util.conv2d(data , mlp[i], [1,1],
                               padding='VALID', stride=[1,1],
                               bn=True, is_training=is_training,
                               scope= name +str(i), bn_decay=bn_decay, is_dist=True)
        else:
            tmp_layer = tf_util.conv2d(layers[-1] , mlp[i], [1,1],
                               padding='VALID', stride=[1,1],
                               bn=True, is_training=is_training,
                               scope= name +str(i), bn_decay=bn_decay, is_dist=True)
        layers.append(tmp_layer)
    return tmp_layer
