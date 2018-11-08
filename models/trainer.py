import numpy as np
import sys
import glob
import pdb
from sklearn.model_selection import train_test_split
import pointnet_sem_seg
basedir = '/media/sanket/My Passport/Sanket/Kitti/training'
sys.path.append('/home/sanket/MS_Thesis/Pointwise-segmentation/kitti_data')
from dataset_iterator import Kitti_data_iterator
import tensorflow as tf



def infer_model(dataset_iterator, model_path, net_out):
    with tf.Session() as sess:
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print ('Model is restored')
        data, label , iter , batch_no= dataset_iterator.get_batch()
        pred = sess.run([net_out], feed_dict = {pcl_placeholder : data,
                                       label_placeholder: label,
                                       is_training_pl:False})
        np.save('data.npy', data)
        np.save('label.npy', label)
        np.save('pred.npy', pred)

def train(dataset_iterator, num_iteration, loss):
    optimizer = tf.train.AdamOptimizer()
    train_op =  optimizer.minimize(loss)
    loss_ar = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data, label , iter , batch_no= dataset_iterator.get_batch()
        while(iter < num_iteration):
            _, batch_loss = sess.run([train_op,loss],feed_dict = {pcl_placeholder : data,
                                           label_placeholder: label,
                                           is_training_pl:True})
            print ("Iter : ", iter , "Batch : " , batch_no ,  "  Loss : ", batch_loss)
            loss_ar.append(batch_loss)
            data, label , iter , batch_no= dataset_iterator.get_batch()
            if ((iter % 10 == 0) and (batch_no == 0)):
                path = "/home/sanket/MS_Thesis/Pointwise-segmentation/saved_model/"
                save_path = saver.save(sess, path + str(iter) +".ckpt")
                print("Model saved in path: %s" % save_path)
        pdb.set_trace()


if __name__=='__main__':
    dataset_iterator = Kitti_data_iterator(basedir, batch_size = 2)
    pcl_placeholder, label_placeholder = pointnet_sem_seg.input_placeholder(batch_size =2,numpoints = 60000)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    net_out = pointnet_sem_seg.get_model(pcl_placeholder, is_training = is_training_pl)
    # pcl_placeholder, label_placeholder = pointnet_sem_seg.input_placeholder()
    loss_model = pointnet_sem_seg.get_loss(net_out, label_placeholder)
    # train(dataset_iterator,num_iteration = 100, loss= loss_model)
    path = "/home/sanket/MS_Thesis/Pointwise-segmentation/saved_model/"
    path += "10.ckpt"
    infer_model(dataset_iterator, path, net_out)
