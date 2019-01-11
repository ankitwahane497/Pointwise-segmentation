import tensorflow as tf
import math
import time
import numpy as np
import os
import pdb
import sys
import tf_utils.edge_tf_util as tf_util
from tf_utils import pointer_util
from sklearn.neighbors import NearestNeighbors
sys.path.append('/home/sanket/MS_Thesis/Pointwise-segmentation/kitti_data')


def input_placeholder(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                   shape=(batch_size, num_point, 4))
    labels_pl = tf.placeholder(tf.float32,
                shape=(batch_size, num_point,8))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)

    k = 30

    global_edge_features = tf.py_func(pointer_util.global_nearest_neighbors_features,[input_image,k],tf.float32)
    local_edge_features  = tf.py_func(pointer_util.local_nearest_neighbors_features,[input_image,k],tf.float32)

    global_edge_features = tf.reshape(global_edge_features, (batch_size,num_point,k,4))
    local_edge_features  = tf.reshape(local_edge_features, (batch_size,num_point,k,4))

    global_feature_1 = pointer_util.feature_network(global_edge_features,
                                                    mlp = [2048,1024],
                                                    name ='global_feature_1_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)
    local_feature_1  = pointer_util.feature_network(local_edge_features,
                                                    mlp = [2048,1024],
                                                    name ='local_feature_1_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)

    out_feature_1 = tf_util.conv2d(tf.concat([global_feature_1, local_feature_1], axis=-1),
                       512, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='out_feature_1', bn_decay=bn_decay, is_dist=True)

    global_edge_features_2 = tf.py_func(pointer_util.global_nearest_neighbors_features,[out_feature_1,k],tf.float32)
    local_edge_features_2  = tf.py_func(pointer_util.local_nearest_neighbors_features,[out_feature_1,k],tf.float32)

    global_edge_features_2 = tf.reshape(global_edge_features_2, (batch_size,num_point,k,4))
    local_edge_features_2  = tf.reshape(local_edge_features_2, (batch_size,num_point,k,4))


    global_feature_2 = pointer_util.feature_network(global_edge_features_2,
                                                    mlp = [512,256],
                                                    name ='global_feature_2_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)
    local_feature_2  = pointer_util.feature_network(local_edge_features_2,
                                                    mlp = [512,256],
                                                    name ='local_feature_2_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)

    out_feature_2 = tf_util.conv2d(tf.concat([global_feature_2, local_feature_2], axis=-1),
                       256, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='out_feature_2', bn_decay=bn_decay, is_dist=True)


    global_edge_features_3 = tf.py_func(pointer_util.global_nearest_neighbors_features,[out_feature_2,k],tf.float32)
    local_edge_features_3  = tf.py_func(pointer_util.local_nearest_neighbors_features,[out_feature_2,k],tf.float32)

    global_edge_features_3 = tf.reshape(global_edge_features_3, (batch_size,num_point,k,4))
    local_edge_features_3  = tf.reshape(local_edge_features_3, (batch_size,num_point,k,4))


    global_feature_3 = pointer_util.feature_network(global_edge_features_3,
                                                    mlp = [256,256],
                                                    name ='global_feature_3_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)
    local_feature_3  = pointer_util.feature_network(local_edge_features_3,
                                                    mlp = [256,256],
                                                    name ='local_feature_3_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)

    out_feature_3 = tf_util.conv2d(tf.concat([global_feature_3, local_feature_3], axis=-1),
                       126, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='out_feature_3', bn_decay=bn_decay, is_dist=True)



    global_edge_features_4 = tf.py_func(pointer_util.global_nearest_neighbors_features,[out_feature_3,k],tf.float32)
    local_edge_features_4  = tf.py_func(pointer_util.local_nearest_neighbors_features,[out_feature_3,k],tf.float32)

    global_edge_features_4 = tf.reshape(global_edge_features_4, (batch_size,num_point,k,4))
    local_edge_features_4  = tf.reshape(local_edge_features_4, (batch_size,num_point,k,4))


    global_feature_4 = pointer_util.feature_network(global_edge_features_4,
                                                    mlp = [126,126],
                                                    name ='global_feature_4_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)
    local_feature_4  = pointer_util.feature_network(local_edge_features_4,
                                                    mlp = [126,126],
                                                    name ='local_feature_4_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)

    out_feature_4 = tf_util.conv2d(tf.concat([global_feature_4, local_feature_4], axis=-1),
                       126, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='out_feature_4', bn_decay=bn_decay, is_dist=True)

    out_max = tf.reduce_max(out_feature_4, axis =-2, keepdims=True)
    net = tf_util.conv2d(out_max, 126, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = tf_util.conv2d(net, 126, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net,8, [1,1], padding='VALID', stride=[1,1],
             activation_fn=None, scope='seg/conv3', is_dist=True)
    net = tf.squeeze(net, [2])
    net_out = tf.nn.softmax(net,axis=-1,name='out')
    return net, net_out
    # return None

def get_loss(pred, label):
  """ pred: B,N,13; label: B,N """
  # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  loss = tf.nn.weighted_cross_entropy_with_logits(targets = label, logits = pred,
            pos_weight = np.array([0.2,3.0,3.0,3.0,1.0,1.0,1.0,1.0]))
  return tf.reduce_mean(loss)


if __name__=='__main__':
    pcl, label = input_placeholder(2,60000)
    # is_train = tf.placeholder(tf.bool, shape= ())
    pcl_h  = np.zeros((2,60000,4))
    is_train = tf.placeholder(tf.bool, shape =())
    model = get_model(pcl, is_train)
    print ('****')
