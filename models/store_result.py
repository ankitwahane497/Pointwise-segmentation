import numpy as np
import sys
import pdb
#basedir = '/media/sanket/My Passport/Sanket/Kitti/training'
sys.path.append('/home/srgujar/Pointwise-segmentation/kitti_data')
# from dataset_iterator import Kitti_data_iterator
from birds_eye_view_projection import birds_eye_view
from clustering_birds_eye_view import clustering_birds_eye_view
import read_kitti_data
import tensorflow as tf
import matplotlib.pyplot as plt


def convert_one_hot_to_label(label):
    return np.argmax(label,axis  = -1)


def unscale_points(pcl):
    pcl[:,:,0] *= 40
    pcl[:,:,1] *= 30
    return pcl



def store_instance_results(data,label_seg,seg_predict,label_instance, instance_prediction, file_path):
    seg_predict = convert_one_hot_to_label(seg_predict)
    pcl  = np.zeros((data.shape[0],data.shape[1],6))
    pcl[:,:,:3] = unscale_points(data[:,:,:3])
    pcl[:,:,3]  = label
    pcl[:,:,4]  = pred
    for i in range(data.shape[0]):
         instance_label = i+1
         pcl[pcl[:,:,4] == 1,5] = instance_label
    #we have a x,x,6 matrix
    pcl = pcl.reshape(data.shape[0]*data.shape[1],6)
    clustering_proj = clustering_birds_eye_view()
    img1, img2 = clustering_proj.get_birds_eye_view(pcl, shift_pcl = False)

    #pdb.set_trace()
    plt.subplot(1,2,1)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(img1)

    plt.subplot(1,2,2)
    plt.title('Pointer')
    plt.axis('off')
    plt.imshow(img2)
    plt.savefig(file_name, dpi = 300)





def store_results(data, label,pred,file_name):
    #pdb.set_trace()
    pred = convert_one_hot_to_label(pred)
    pcl  = np.zeros((data.shape[0],data.shape[1],6))
    pcl[:,:,:3] = unscale_points(data[:,:,:3])
    pcl[:,:,3]  = label
    pcl[:,:,4]  = pred
    for i in range(data.shape[0]):
         instance_label = i+1
         pcl[pcl[:,:,4] == 1,5] = instance_label
    #we have a x,x,6 matrix
    pcl = pcl.reshape(data.shape[0]*data.shape[1],6)
    clustering_proj = clustering_birds_eye_view()
    img1, img2 = clustering_proj.get_birds_eye_view(pcl, shift_pcl = False)

    #pdb.set_trace()
    plt.subplot(1,2,1)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(img1)

    plt.subplot(1,2,2)
    plt.title('Pointer')
    plt.axis('off')
    plt.imshow(img2)
    plt.savefig(file_name, dpi = 300)


def store_results_with_cam(data, label,pred,cam_image,file_name):
    pdb.set_trace()
    pred = convert_one_hot_to_label(pred)
    data  = unscale_points(data[0])
    #pdb.set_trace()

    plt.subplot(1,3,1)
    plt.title('Camera Image')
    plt.axis('off')
    plt.imshow(cam_image)


    bp = birds_eye_view()
    c1  = np.reshape(label[0],(10000,1))
    pcl = np.hstack((data,c1))
    p1 = bp.get_birds_eye_view(pcl)

    plt.subplot(1,3,2)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(p1)

    bp1 = birds_eye_view()
    c2  = np.reshape(pred[0],(10000,1))
    pcl2 = np.hstack((data,c2))
    p2 = bp1.get_birds_eye_view(pcl2)
    plt.subplot(1,3,3)
    plt.title('Pointer Predictions')
    plt.axis('off')
    plt.imshow(p2)

    plt.savefig(file_name, dpi = 300)
