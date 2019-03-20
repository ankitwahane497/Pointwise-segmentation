import tensorflow as tf
import math
import time
import numpy as np
import os
import pdb
import sys
sys.path.append('/home/srgujar/Pointwise-segmentation/models/tf_utils')
import tf_utils.edge_tf_util as tf_util
from tf_utils import pointer_util
from sklearn.neighbors import NearestNeighbors
sys.path.append('/home/srgujar/Pointwise-segmentation/kitti_data')


def input_placeholder(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                   shape=(batch_size, num_point, 4),name = 'pcl_placeholder')
    instance_pl  = tf.placeholder(tf.float32, shape = (batch_size,num_point, 3) , name = 'instance_placeholder')
    labels_pl = tf.placeholder(tf.float32,
                shape=(batch_size, num_point,2),name ='label_placeholder')
    print ('********Training Instance Model 1***************************')
    return pointclouds_pl, instance_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -2)

    k = 10
    nearest_pts_id = tf.py_func(pointer_util.get_nearest_neighbors_id,[input_image,k],tf.int32)
    # pointer_util.get_nearest_neighbors_id(input_image,k)
    # pdb.set_trace()
    nearest_pts_id = tf.reshape(nearest_pts_id, (batch_size, num_point,k))
    #pdb.set_trace()

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
    out_feature_1 = tf.concat([input_image, out_feature_1],axis = 3)
    global_edge_features_2 = tf.py_func(pointer_util.get_global_features,[out_feature_1,nearest_pts_id],tf.float32)
    local_edge_features_2  = tf.py_func(pointer_util.get_local_features,[out_feature_1,nearest_pts_id],tf.float32)
    
    #pdb.set_trace()
    global_edge_features_2 = tf.reshape(global_edge_features_2, (batch_size,num_point,k,130))
    local_edge_features_2  = tf.reshape(local_edge_features_2, (batch_size,num_point,k,130))
    
    
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
    
    #out_feature_2 = tf_util.conv2d(global_feature_2,
    #                   256, [1,1],
    #                   padding='VALID', stride=[1,1],
    #                   bn=True, is_training=is_training,
    #                   scope='out_feature_2', bn_decay=bn_decay, is_dist=True)
    
    out_feature_2 = tf.reduce_max(out_feature_2, axis = -2, keepdims = True)
    out_feature_2 = tf.concat([out_feature_1,out_feature_2],axis = 3)    

    global_edge_features_3 = tf.py_func(pointer_util.get_global_features,[out_feature_2,nearest_pts_id],tf.float32)
    local_edge_features_3  = tf.py_func(pointer_util.get_local_features,[out_feature_2,nearest_pts_id],tf.float32)
    #pdb.set_trace()
    global_edge_features_3 = tf.reshape(global_edge_features_3, (batch_size,num_point,k,386))
    local_edge_features_3  = tf.reshape(local_edge_features_3, (batch_size,num_point,k,386))
    
    
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
    
    out_feature_3 = tf.reduce_max(out_feature_3, axis = -2, keepdims = True)
    out_feature_3 = tf.concat([out_feature_2,out_feature_3] , axis = 3)
    
    global_edge_features_4 = tf.py_func(pointer_util.get_global_features,[out_feature_3,nearest_pts_id],tf.float32)
    local_edge_features_4  = tf.py_func(pointer_util.get_local_features,[out_feature_3,nearest_pts_id],tf.float32)
    #pdb.set_trace()
    global_edge_features_4 = tf.reshape(global_edge_features_4, (batch_size,num_point,k,512))
    local_edge_features_4  = tf.reshape(local_edge_features_4, (batch_size,num_point,k,512))
    
    
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

    #out_max = tf.reduce_max(global_feature_1, axis =-2, keepdims=True)
    # out_max = tf.reduce_max(out_feature_1, axis =-2, keepdims=True)
    #out_feature_4 = tf.reduce_max(out_feature_4, axis =-2, keepdims=True)
    #out_max = tf.concat([out_feature_3,out_feature_4], axis = 3)
    out_max =out_feature_2
    net = tf_util.conv2d(out_max, 512, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1],
              bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    net1 = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
              bn=True, is_training=is_training, scope='seg/conv3', is_dist=True)
    net1 = tf_util.dropout(net1, keep_prob=0.5, is_training=is_training, scope='dp1')
    net1 = tf_util.conv2d(net1,2, [1,1], padding='VALID', stride=[1,1],
             activation_fn=None, scope='seg/conv4', is_dist=True)
    net1 = tf.squeeze(net1,[2],name ='net_pred')
    
    net2 = tf_util.conv2d(net, 256, [1,1] ,padding = 'VALID' , stride = [1,1] , bn= True, is_training = is_training, scope = 'inst/conv1', is_dist = True)
    net2 = tf_util.conv2d(net2,126,[1,1] , padding = 'VALID', stride = [1,1], bn =True, is_training = is_training, scope  = 'inst/conv2', is_dist = True)
    net2 = tf_util.conv2d(net2,3,[1,1] , padding = 'VALID', stride = [1,1] , bn =True, is_training = is_training, scope = 'inst/conv3', is_dist = True)
    net2 = tf.squeeze(net2, [2], name = 'instance_pred')

    #pdb.set_trace()
    net_out = tf.nn.softmax(net1,axis=-1,name='net_pred_softmax')
    return net1,net2, net_out

def get_loss(pred_seg,label_seg,pred_instance, label_instance):
  """ pred: B,N,13; label: B,N """
  # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  seg_loss = tf.nn.weighted_cross_entropy_with_logits(targets = label_seg, logits = pred_seg,
            pos_weight = np.array([1.0,60.0]))
  instance_loss  = tf.losses.cosine_distance(tf.nn.l2_normalize(pred_instance, 0), tf.nn.l2_normalize(label_instance, 0), dim=0)
  total_loss = tf.reduce_mean(seg_loss) + tf.reduce_mean(instance_loss)
  return total_loss


if __name__=='__main__':
    pcl, label = input_placeholder(2,60000)
    # is_train = tf.placeholder(tf.bool, shape= ())
    pcl_h  = np.zeros((2,60000,4))
    is_train = tf.placeholder(tf.bool, shape =())
    model = get_model(pcl, is_train)
    pdb.set_trace()
    print('End log')
