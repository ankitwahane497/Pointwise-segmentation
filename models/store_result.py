import numpy as np
import sys
import pdb
#basedir = '/media/sanket/My Passport/Sanket/Kitti/training'
sys.path.append('/home/srgujar/Pointwise-segmentation/kitti_data')
# from dataset_iterator import Kitti_data_iterator
from birds_eye_view_projection import birds_eye_view
import read_kitti_data
import tensorflow as tf
import matplotlib.pyplot as plt


def convert_one_hot_to_label(label):
    return np.argmax(label,axis  = -1)


def unscale_points(pcl):
    pcl[:,0] *= 40
    pcl[:,1] *= 30
    return pcl



def store_results(data, label,pred,file_name):
    pred = convert_one_hot_to_label(pred)
    data  = unscale_points(data[0])
    #pdb.set_trace()
    bp = birds_eye_view()
    c1  = np.reshape(label[0],(10000,1))
    pcl = np.hstack((data,c1))
    p1 = bp.get_birds_eye_view(pcl)
    plt.subplot(1,2,1)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(p1)

    bp1 = birds_eye_view()
    c2  = np.reshape(pred[0],(10000,1))
    pcl2 = np.hstack((data,c2))
    p2 = bp1.get_birds_eye_view(pcl2)
    plt.subplot(1,2,2)
    plt.title('Pointer')
    plt.axis('off')
    plt.imshow(p2)
    plt.savefig(file_name, dpi = 300)
