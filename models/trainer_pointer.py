import numpy as np
import sys
import glob
import pdb
from sklearn.model_selection import train_test_split
sys.path.append('/home/srgujar/Pointwise-segmentation/models/tf_utils')
import pointer_instance as model
basedir ='/home/srgujar/kitti'
sys.path.append('/home/srgujar/Pointwise-segmentation/kitti_data')
from dataset_iterator import Kitti_data_iterator
basedir ='/home/srgujar/kitti'
basedir_testing  ='/home/srgujar/Data/testing'
import tensorflow as tf
import logging
import os
from result_dir import *
from store_result import *


def infer_model_trained(dataset_iterator, model_path, net_out,save_model_path):
    with tf.Session() as sess:
        # tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print ('Model is restored :', model_path)
        data, label , cam_images, iter , batch_no= dataset_iterator.get_batch_with_images()
        label = make_new_class(label)
        label_ = get_one_hot_label(label)
        counter = 0
        try:
            os.makedirs(save_model_path + '/result/validation')
        except:
            print ('File already exists')
        while (iter == 0):
            pred = sess.run(net_out, feed_dict = {pcl_placeholder : data,
                                       is_training_pl:False})
            a_2 = calculate_accuracy(pred, label)
            a_3 = calculate_class_accuracy(pred, label)
            a_4 = calculate_car_accuracy(pred,label)
            file_path = save_model_path + '/result/validation/' + str(counter) + '.png'
            store_results_with_cam(data,label,pred,cam_images[0], file_path)
            np.save(save_model_path + '/result/validation/data' + str(counter) + '.npy', data)
            np.save(save_model_path + '/result/validation/label'+ str(counter) + '.npy', label)
            np.save(save_model_path + '/result/validation/pred' + str(counter) + '.npy', pred)
            print ('saved prediction of ' + str(counter) + ' accuracy : ',a_2 , ' class accuracy : ',a_3,  ' car_class_accuracy : ' ,a_4)
            data, label,  cam_images, iter , batch_no = dataset_iterator.get_batch_with_images()
            label = make_new_class(label)
            label_ = get_one_hot_label(label)
            counter += 1


def infer_model(dataset_iterator,sess, net_out,save_model_path,iteration, num_samples = 10):
    data, label_instance ,label_seg, iteration_num , batch_no= dataset_iterator.get_batch()
    label_seg = make_new_class(label_seg)
    label_seg_ = get_one_hot_label(label_seg)
    counter = 0
    os.makedirs(save_model_path +'/result/epoch_' +str(iteration))
    #os.makedirs(save_model_path +'/result/raw')
    while (counter < num_samples):
        seg_predict,instance_prediction = sess.run([seg_pred, instance_pred],feed_dict = {pcl_placeholder : data,
                                       is_training_pl:False})
        a_2 = calculate_accuracy(seg_predict, label_seg)*100
        a_3 = calculate_class_accuracy(seg_predict, label_seg)*100
        a_4 = calculate_car_accuracy(seg_predict,label_seg)*100
        file_path = save_model_path + '/result/epoch_'+str(iteration) + '/' + str(counter) + '.png'
        # store_instance_results(data,label_seg,seg_predict,label_instance, instance_prediction, file_path)
        store_results(data,label_seg,seg_predict,file_path)
        np.save(save_model_path + '/result/epoch_'+str(iteration) +'/data' + str(counter) + '.npy', data)
        np.save(save_model_path + '/result/epoch_' +str(iteration) +'/label_seg'+ str(counter) + '.npy', label_seg)
        np.save(save_model_path + '/result/epoch_' + str(iteration) + '/pred_seg' +str(counter) + '.npy', seg_predict)
        np.save(save_model_path + '/result/epoch_' +str(iteration) +'/label_instance'+ str(counter) + '.npy', label_instance)
        np.save(save_model_path + '/result/epoch_' + str(iteration) + '/pred_instance' +str(counter) + '.npy', instance_prediction)
        print ('saved prediction of ' + str(counter) + ' accuracy : ',a_2 , ' class accuracy : ',a_3,  ' car_class_accuracy : ' ,a_4)
        data, label_instance ,label_seg, iteration_num , batch_no= dataset_iterator.get_batch()
        label_seg = make_new_class(label_seg)
        label_seg_ = get_one_hot_label(label_seg)
        counter += 1


def calculate_accuracy(prediction, labels):
    #pdb.set_trace()
    prediction = np.argmax(prediction, axis = -1 )
    correct = np.sum(prediction == labels)
    #pdb.set_trace()
    return (correct/ (len(prediction)*len(prediction[0])))

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
        return (np.mean(c2)/10.)
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
        correct = (correct/ (len(c2)))
        return correct/10.


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

def train(dataset_iterator,test_iter, num_iteration, loss, seg_pred, instance_pred):
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
        data, label_instance ,label_seg, iteration_num , batch_no= dataset_iterator.get_batch()
        label_seg = make_new_class(label_seg)
        label_seg_ = get_one_hot_label(label_seg)
        #pdb.set_trace()
        while(iteration_num< num_iteration):
            _, batch_loss,seg_predict,instance_prediction = sess.run([train_op,loss, seg_pred, instance_pred],feed_dict = {pcl_placeholder : data,
                                           seg_label_placeholder: label_seg_,
                                           instance_label_placeholder: label_instance,
                                           is_training_pl:True})

            accuracy = calculate_accuracy(seg_predict, label_seg)*100
            class_accuracy = calculate_class_accuracy(seg_predict, label_seg)*100
            car_acc = calculate_car_accuracy(seg_predict,label_seg)*100
            print ("Iter : ", iteration_num , "Batch : " , batch_no ,  "  Loss : ", batch_loss ,
                    " Accuracy : ",accuracy, " Class Accuracy : ", class_accuracy , " Car class accuracy " , car_acc )
            log = "Iter : " + str(iteration_num) + " Batch : " + str(batch_no) ,  "  Loss : " + str(batch_loss) + " Accuracy : " + str(accuracy) +  " Class Accuracy : "+ str(class_accuracy)
            logging.info(log)
            loss_ar.append(batch_loss)
            acc_all.append(accuracy)
            class_acc.append(class_accuracy)

            #print('Instance passed')
            data, label_instance ,label_seg, iteration_num , batch_no= dataset_iterator.get_batch()
            label_seg  = make_new_class(label_seg)
            label_seg_ = get_one_hot_label(label_seg)
            #pdb.set_trace()

            if(batch_no == 0):
                batch_accuracy = np.mean(acc_all)
                class_accuracy = np.mean(class_acc)
                batch_loss_mean= np.mean(loss_ar)
                log = "**** Iteration : " +  str(iteration_num) + " loss : " + str(batch_loss_mean) + " Accuracy: " + str(batch_accuracy) +" Class Accuracy : " + str(class_accuracy)
                logging.info(log)
                print (log)

            if ((iteration_num % 10  == 0)and (batch_no == 0)):
                path = result_repo + '/checkpoints/instance_pointer2__'
                save_path = saver.save(sess, path +str(iteration_num) +"_"+ str(batch_no) +".ckpt")
                print("Model saved in path: %s" % save_path)
                infer_model(test_iter,sess, pred, result_repo , iteration_num , num_samples = 14)

        return result_repo





if __name__=='__main__':
    dataset_iterator = Kitti_data_iterator(basedir, batch_size = 1, num_points = 10000)
    dataset_iterator_test = Kitti_data_iterator(basedir, batch_size = 1, num_points = 10000)
    pcl_placeholder, instance_label_placeholder, seg_label_placeholder  = model.input_placeholder(batch_size =1,num_point = 10000)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    net_seg_out, net_instance_out, net_seg_softmax = model.get_model(pcl_placeholder, is_training = is_training_pl)
    loss_model = model.get_loss(net_seg_out,seg_label_placeholder, net_instance_out, instance_label_placeholder)
    result_repo = train(dataset_iterator,dataset_iterator_test,num_iteration = 200, loss= loss_model, seg_pred= net_seg_softmax, instance_pred = net_instance_out)
    #path  = "/home/srgujar/Pointwise-segmentation/results/pointer_M2_2_1_11_57"
    #model_path = path +  "/checkpoints/pointer2__3_0.ckpt"
    #infer_model_trained(dataset_iterator_test, model_path, net_pred,path)
