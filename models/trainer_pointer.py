import numpy as np
import sys
import glob
import pdb
from sklearn.model_selection import train_test_split
sys.path.append('/home/srgujar/Pointwise-segmentation/models/tf_utils')
import pointer_sem_seg_4 as model
# basedir = '/media/sanket/My Passport/Sanket/Kitti/training'
#basedir = '/home/sanket/MS_Thesis/kitti'
#basedir ='/home/srgujar/Data/training'
basedir ='/home/srgujar/kitti'
sys.path.append('/home/srgujar/Pointwise-segmentation/kitti_data')
from dataset_iterator import Kitti_data_iterator
import tensorflow as tf
import logging
import os
from result_dir import *
from store_result import *
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
#
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


def infer_model_trained(dataset_iterator, model_path, net_out,save_model_path):
    with tf.Session() as sess:
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print ('Model is restored :', model_path)
        data, label , iter , batch_no= dataset_iterator.get_batch()
        label_ = get_one_hot_label(label)
        label = make_new_class(label)
        counter = 0
        while (iter == 0):
            pred = sess.run(net_out, feed_dict = {pcl_placeholder : data,
                                       label_placeholder: label_,
                                       is_training_pl:False})
            a_2 = calculate_accuracy(pred, label)
            a_3 = calculate_class_accuracy(pred, label)
            a_4 = calculate_car_accuracy(pred,label)
            np.save(save_model_path + '/result/data' + str(counter) + '.npy', data)
            np.save(save_model_path + '/result/label'+ str(counter) + '.npy', label)
            np.save(save_model_path + '/result/pred' + str(counter) + '.npy', pred)
            print ('saved prediction of ' + str(counter) + ' accuracy : ',a_2 , ' class accuracy : ',a_3,  ' car_class_accuracy : ' ,a_4)
            data, label, iter , batch_no = dataset_iterator.get_batch()
            label_ = get_one_hot_label(label)
            label = make_new_class(label)
            counter += 1


def infer_model(dataset_iterator,sess, net_out,save_model_path,iteration, num_samples = 10):
    data, label , iter , batch_no= dataset_iterator.get_batch()
    label = make_new_class(label)
    label_ = get_one_hot_label(label)
    counter = 0
    os.makedirs(save_model_path +'/result/epoch_' +str(iteration))
    #os.makedirs(save_model_path +'/result/raw')
    while (counter < num_samples):
        pred = sess.run(net_out, feed_dict = {pcl_placeholder : data,
                                       label_placeholder: label_,
                                       is_training_pl:False})
        a_2 = calculate_accuracy(pred, label)
        a_3 = calculate_class_accuracy(pred, label)
        a_4 = calculate_car_accuracy(pred,label)
        file_path = save_model_path + '/result/epoch_'+str(iteration) + '/' + str(counter) + '.png'
        store_results(data,label,pred,file_path)
        np.save(save_model_path + '/result/epoch_'+str(iteration) +'/data' + str(counter) + '.npy', data)
        np.save(save_model_path + '/result/epoch_' +str(iteration) +'/label'+ str(counter) + '.npy', label)
        np.save(save_model_path + '/result/epoch_' + str(iteration) + '/pred' +str(counter) + '.npy', pred)
        print ('saved prediction of ' + str(counter) + ' accuracy : ',a_2 , ' class accuracy : ',a_3,  ' car_class_accuracy : ' ,a_4)
        data, label, iter , batch_no = dataset_iterator.get_batch()
        label = make_new_class(label)
        label_ = get_one_hot_label(label)
        counter += 1


def calculate_accuracy(prediction, labels):
    #pdb.set_trace()
    prediction = np.argmax(prediction, axis = -1 )
    correct = np.sum(prediction == labels)
    #pdb.set_trace()
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
        if len(indx2) ==  0:
            pass
        else:
            correct = np.sum(np.in1d(indx1,indx2))
            correct = (correct/ len(indx2))
            c2.append(correct)
    if len(c2) > 0 : #checking for empty array for no class frame     
        return np.mean(c2)
    else:
        return 0.


def calculate_car_accuracy(pred, label):
    pred = np.argmax(pred, axis = -1)
    c1 = np.where(pred == 1)[1]
    c2 = np.where(label == 1)[1]
    if len(c2) == 0:
        return 0.
    else:
        correct = np.sum(np.in1d(c1,c2))
        correct = (correct/ len(c2))
        return correct


def get_one_hot_label(label):
    shape_l = label.shape
    one_hot = np.zeros((shape_l[0],shape_l[1],2))
    for j in range(shape_l[0]):
        for i in range(shape_l[1]):
            try:
                one_hot[j][i][int(label[j][i])] = 1
            except:
                pdb.set_trace()
    return one_hot

def make_new_class(label):
    shape_label = label.shape
    new_label = np.zeros((shape_label[0], shape_label[1]))
    for i in range(shape_label[0]):
        for j in range(shape_label[1]):
            if label[i][j] == 0:
                new_label[i][j] = 0
            else:
                new_label[i][j] = 1
    return new_label

def train(dataset_iterator,test_iter, num_iteration, loss, pred):
    optimizer = tf.train.AdamOptimizer()
    train_op =  optimizer.minimize(loss)
    result_repo = make_result_def('/home/srgujar/Pointwise-segmentation/results','pointer_M2')
    logging.basicConfig(level=logging.DEBUG, filename=result_repo + "/log/log.txt", filemode="a+",
                        format="%(asctime)-15s %(message)s")
    loss_ar = []
    acc_all = []
    class_acc = []
    #result_repo = make_result_def('/home/srgujar/Pointwise-segmentation/results','pointer_M2')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data, label , iter , batch_no= dataset_iterator.get_batch()
        label = make_new_class(label)
        label_ = get_one_hot_label(label)
        while(iter < num_iteration):
            _, batch_loss, predict = sess.run([train_op,loss, pred],feed_dict = {pcl_placeholder : data,
                                           label_placeholder: label_,
                                           is_training_pl:True})
            accuracy = calculate_accuracy(predict, label)*100
            class_accuracy = calculate_class_accuracy(predict, label)*100
            car_acc = calculate_car_accuracy(predict,label)*100
            print ("Iter : ", iter , "Batch : " , batch_no ,  "  Loss : ", batch_loss ,
                    " Accuracy : ",accuracy, " Class Accuracy : ", class_accuracy , " Car class accuracy " , car_acc )
            log = "Iter : " + str(iter) + " Batch : " + str(batch_no) ,  "  Loss : " + str(batch_loss) + " Accuracy : " + str(accuracy) +  " Class Accuracy : "+ str(class_accuracy)
            logging.info(log)
            loss_ar.append(batch_loss)
            acc_all.append(accuracy)
            class_acc.append(class_accuracy)
            data, label , iter , batch_no= dataset_iterator.get_batch()
            label  = make_new_class(label)
            label_ = get_one_hot_label(label)

            if(batch_no == 0):
                batch_accuracy = np.mean(acc_all)
                class_accuracy = np.mean(class_acc)
                batch_loss_mean= np.mean(loss_ar)
                log = "**** Iteration : " +  str(iter) + " loss : " + str(batch_loss_mean) + " Accuracy: " + str(batch_accuracy) +" Class Accuracy : " + str(class_accuracy) 
                logging.info(log)
                print (log)
 
            if ((iter % 10  == 0)and (batch_no == 0)):
                path = result_repo + '/checkpoints/pointer2__'
                save_path = saver.save(sess, path +str(iter) +"_"+ str(batch_no) +".ckpt")
                print("Model saved in path: %s" % save_path)
                infer_model(test_iter,sess, pred, result_repo , iter, num_samples = 10)
        return result_repo


if __name__=='__main__':
    dataset_iterator = Kitti_data_iterator(basedir, batch_size = 1, num_points = 10000)
    dataset_iterator_test = Kitti_data_iterator(basedir, batch_size = 1, num_points = 10000)
    pcl_placeholder, label_placeholder = model.input_placeholder(batch_size =1,num_point = 10000)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    net_out, net_pred = model.get_model(pcl_placeholder, is_training = is_training_pl)
    loss_model = model.get_loss(net_out, label_placeholder)
    result_repo = train(dataset_iterator,dataset_iterator_test,num_iteration = 200, loss= loss_model, pred= net_pred)
    #path  = "/home/srgujar/Pointwise-segmentation/results/pointer_M3_1_19_15_36"
    #model_path = path +  "/checkpoints/pointer3_50_0.ckpt"
    #infer_model(dataset_iterator, model_path, net_pred,path)
