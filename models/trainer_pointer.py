import numpy as np
import sys
import glob
import pdb
from sklearn.model_selection import train_test_split
import pointer_sem_seg_2 as model
# basedir = '/media/sanket/My Passport/Sanket/Kitti/training'
basedir = '/home/sanket/MS_Thesis/kitti'
sys.path.append('/home/sanket/MS_Thesis/Pointwise-segmentation/kitti_data')
from dataset_iterator import Kitti_data_iterator
import tensorflow as tf
import logging
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
#
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


def infer_model(dataset_iterator, model_path, net_out):
    with tf.Session() as sess:
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print ('Model is restored')
        data, label , iter , batch_no= dataset_iterator.get_batch()
        label_ = get_one_hot_label(label)
        pred = sess.run([net_out], feed_dict = {pcl_placeholder : data,
                                       label_placeholder: label_,
                                       is_training_pl:False})
        np.save('data1.npy', data)
        np.save('label1.npy', label)
        np.save('pred1.npy', pred)


def calculate_accuracy(prediction, labels):
    prediction = np.argmax(prediction, axis = -1 )
    correct = np.sum(prediction == labels)
    return (correct/ len(prediction[0]))

def calculate_class_accuracy(prediction, labels):
    prediction = np.argmax(prediction, axis = -1 )
    key_dict = {'Car':1, 'Van': 2, 'Truck':3,
             'Pedestrian':4, 'Person_sitting':5,
             'Cyclist': 6 , 'Tram' : 7 ,
             'Misc' : 0 , 'DontCare': 0}
    c2 = []
    for i in range(1,8):
        indx1 =  np.where(prediction == i)[1]
        indx2 =  np.where(labels  == i)[1]
        # pdb.set_trace()
        if len(indx2) ==  0:
            return 0
        correct = np.sum(np.in1d(indx1,indx2))
        correct = (correct/ len(indx2))
        c2.append(correct)
    return np.mean(c2)

def get_one_hot_label(label):
    shape_l = label.shape
    one_hot = np.zeros((shape_l[0],shape_l[1],8))
    for j in range(shape_l[0]):
        for i in range(shape_l[1]):
            try:
                one_hot[j][i][int(label[j][i])] = 1
            except:
                pdb.set_trace()
    return one_hot

def train(dataset_iterator, num_iteration, loss, pred):
    optimizer = tf.train.AdamOptimizer()
    train_op =  optimizer.minimize(loss)
    logging.basicConfig(level=logging.DEBUG, filename="edge_conv_new.txt", filemode="a+",
                        format="%(asctime)-15s %(message)s")
    loss_ar = []
    acc_all = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data, label , iter , batch_no= dataset_iterator.get_batch()
        label_ = get_one_hot_label(label)
        while(iter < num_iteration):
            _, batch_loss, predict = sess.run([train_op,loss, pred],feed_dict = {pcl_placeholder : data,
                                           label_placeholder: label_,
                                           is_training_pl:True})
            accuracy = calculate_accuracy(predict, label)*100
            class_accuracy = calculate_class_accuracy(predict, label)*100
            print ("Iter : ", iter , "Batch : " , batch_no ,  "  Loss : ", batch_loss ,
                    " Accuracy : ",accuracy, " Class Accuracy : ", class_accuracy  )
            log = "Iter : " + str(iter) + " Batch : " + str(batch_no) ,  "  Loss : " + str(batch_loss) + " Accuracy : " + str(accuracy) +  " Class Accuracy : "+ str(class_accuracy)
            logging.info(log)
            loss_ar.append(batch_loss)
            acc_all.append(accuracy)
            # pdb.set_trace()
            data, label , iter , batch_no= dataset_iterator.get_batch()
            label_ = get_one_hot_label(label)
            if ((iter % 3 == 0) and (batch_no == 0)):
                path = "/home/sanket/MS_Thesis/Pointwise-segmentation/saved_model/edge_conv_new_"
                save_path = saver.save(sess, path + str(iter) +".ckpt")
                print("Model saved in path: %s" % save_path)
        pdb.set_trace()


if __name__=='__main__':
    dataset_iterator = Kitti_data_iterator(basedir, batch_size = 1, num_points = 10000)
    pcl_placeholder, label_placeholder = model.input_placeholder(batch_size =1,num_point = 10000)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    net_out, net_pred = model.get_model(pcl_placeholder, is_training = is_training_pl)
    loss_model = model.get_loss(net_pred, label_placeholder)
    train(dataset_iterator,num_iteration = 20, loss= loss_model, pred= net_pred)
    # path = "/home/sanket/MS_Thesis/Pointwise-segmentation/saved_model/"
    # path += "edge_conv_new_3.ckpt"
    # infer_model(dataset_iterator, path, net_pred)
