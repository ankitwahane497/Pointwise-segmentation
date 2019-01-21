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
    labels_pl = tf.placeholder(tf.int32,
                shape=(batch_size, num_point))
    print ('*****Training_model_3*********')
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """net :without softmax, net_out: with_softmax"""
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -2)

    k = 30
    nearest_pts_id = tf.py_func(pointer_util.get_nearest_neighbors_id,[input_image,k],tf.int32)
    nearest_pts_id = tf.reshape(nearest_pts_id, (num_point,k))

    global_edge_features = tf.py_func(pointer_util.get_global_features,[input_image,nearest_pts_id],tf.float32)
    local_edge_features  = tf.py_func(pointer_util.get_local_features,[input_image,nearest_pts_id],tf.float32)

    global_edge_features = tf.reshape(global_edge_features, (batch_size,num_point,k,4))
    local_edge_features  = tf.reshape(local_edge_features, (batch_size,num_point,k,4))

    global_feature_1 = pointer_util.feature_network(global_edge_features,
                                                    mlp = [126],
                                                    name ='global_feature_1_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)
    local_feature_1  = pointer_util.feature_network(local_edge_features,
                                                    mlp = [126],
                                                    name ='local_feature_1_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)

    out_feature_1 = tf_util.conv2d(tf.concat([global_feature_1, local_feature_1], axis=-1),
                       126, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='out_feature_1', bn_decay=bn_decay, is_dist=True)

    out_feature_1 = tf.reduce_max(out_feature_1, axis = -2, keepdims = True)
    global_edge_features = tf.reduce_max(global_edge_features, axis = -2, keepdims = True)
    global_edge_features_2 = tf.py_func(pointer_util.get_global_features,[out_feature_1,nearest_pts_id],tf.float32)
    local_edge_features_2  = tf.py_func(pointer_util.get_local_features,[out_feature_1,nearest_pts_id],tf.float32)

    global_edge_features_2 = tf.reshape(global_edge_features_2, (batch_size,num_point,k,126))
    local_edge_features_2  = tf.reshape(local_edge_features_2, (batch_size,num_point,k,126))


    global_feature_2 = pointer_util.feature_network(global_edge_features_2,
                                                    mlp = [256],
                                                    name ='global_feature_2_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)
    local_feature_2  = pointer_util.feature_network(local_edge_features_2,
                                                    mlp = [256],
                                                    name ='local_feature_2_',
                                                    is_training = is_training,
                                                    bn_decay = bn_decay)

    out_feature_2 = tf_util.conv2d(tf.concat([global_feature_2, local_feature_2], axis=-1),
                       256, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='out_feature_2', bn_decay=bn_decay, is_dist=True)
    global_edge_features_2 = tf.reduce_max(global_edge_features_2, axis = -2 , keepdims = True)
    out_feature_2 = tf.reduce_max(out_feature_2, axis = -2, keepdims = True)

    # global_edge_features_3 = tf.py_func(pointer_util.get_global_features_deep,[out_feature_2,nearest_pts_id],tf.float32)
    # local_edge_features_3  = tf.py_func(pointer_util.get_local_features_deep,[out_feature_2,nearest_pts_id],tf.float32)
    #
    # global_edge_features_3 = tf.reshape(global_edge_features_3, (batch_size,num_point,k,256))
    # local_edge_features_3  = tf.reshape(local_edge_features_3, (batch_size,num_point,k,256))
    #
    #
    # global_feature_3 = pointer_util.feature_network(global_edge_features_3,
    #                                                 mlp = [256,256],
    #                                                 name ='global_feature_3_',
    #                                                 is_training = is_training,
    #                                                 bn_decay = bn_decay)
    # local_feature_3  = pointer_util.feature_network(local_edge_features_3,
    #                                                 mlp = [256,256],
    #                                                 name ='local_feature_3_',
    #                                                 is_training = is_training,
    #                                                 bn_decay = bn_decay)
    #
    # out_feature_3 = tf_util.conv2d(tf.concat([global_feature_3, local_feature_3], axis=-1),
    #                    126, [1,1],
    #                    padding='VALID', stride=[1,1],
    #                    bn=True, is_training=is_training,
    #                    scope='out_feature_3', bn_decay=bn_decay, is_dist=True)
    #
    #
    #
    # global_edge_features_4 = tf.py_func(pointer_util.get_global_features_deep,[out_feature_3,nearest_pts_id],tf.float32)
    # local_edge_features_4  = tf.py_func(pointer_util.get_local_features_deep,[out_feature_3,nearest_pts_id],tf.float32)
    # #pdb.set_trace()
    # global_edge_features_4 = tf.reshape(global_edge_features_4, (batch_size,num_point,k,126))
    # local_edge_features_4  = tf.reshape(local_edge_features_4, (batch_size,num_point,k,126))
    #
    #
    # global_feature_4 = pointer_util.feature_network(global_edge_features_4,
    #                                                 mlp = [126,126],
    #                                                 name ='global_feature_4_',
    #                                                 is_training = is_training,
    #                                                 bn_decay = bn_decay)
    # local_feature_4  = pointer_util.feature_network(local_edge_features_4,
    #                                                 mlp = [126,126],
    #                                                 name ='local_feature_4_',
    #                                                 is_training = is_training,
    #                                                 bn_decay = bn_decay)
    #
    # out_feature_4 = tf_util.conv2d(tf.concat([global_feature_4, local_feature_4], axis=-1),
    #                    126, [1,1],
    #                    padding='VALID', stride=[1,1],
    #                    bn=True, is_training=is_training,
    #                    scope='out_feature_4', bn_decay=bn_decay, is_dist=True)
    #
    # out_max = tf.reduce_max(out_feature_4, axis =-2, keepdims=True)
    net  = tf_util.conv2d(tf.concat([global_edge_features, global_edge_features_2,out_feature_1,out_feature_2], axis=-1),
                       1024, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='aggregation', bn_decay=bn_decay, is_dist=True)
    out_max = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
    expand = tf.tile(out_max, [1, num_point, 1, 1])
    net = tf.concat([expand, global_edge_features, global_edge_features_2],axis = 3)
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = tf_util.conv2d(net, 126, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    net = tf_util.conv2d(net, 126, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv3', is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net,2, [1,1], padding='VALID', stride=[1,1],
             activation_fn=None, scope='seg/conv4', is_dist=True)
    net = tf.squeeze(net, [2])
    net_out = tf.nn.softmax(net,axis=-1,name='out')
    #pdb.set_trace()
    return net, net_out

def get_loss(pred, label):
  """ pred: B,N,13; label: B,N """
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  #loss = tf.nn.weighted_cross_entropy_with_logits(targets = label, logits = pred,
  #          pos_weight = np.array([0.2,50,40.0,50.0,50.0,1.0,1.0,1.0]))
  #pdb.set_trace()
  #car_loss = get_car_class_loss(pred, label)
  return tf.reduce_mean(loss) 

def get_car_class_loss(pred,label):
    pred  = tf.argmax(pred, axis = -1)
    c1 = tf.equal(label,1)
    c2 = tf.equal(pred,1)
    if c1.shape[-1] == 0:
        return  0.
    else:
        return (1 / tf.divide(tf.reduce_sum(tf.cast(tf.equal(c1,c2),tf.float32)),c1.shape[-1]))


if __name__=='__main__':
    pcl, label = input_placeholder(2,10000)
    pcl_h  = np.zeros((2,10000,3))
    is_train = tf.placeholder(tf.bool, shape =())
    model = get_model(pcl, is_train)
