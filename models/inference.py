import numpy as np
import sys
import pdb
basedir = '/media/sanket/My Passport/Sanket/Kitti/training'
sys.path.append('/home/sanket/MS_Thesis/Pointwise-segmentation/kitti_data')
# from dataset_iterator import Kitti_data_iterator
from birds_eye_view_projection import birds_eye_view
import read_kitti_data
import tensorflow as tf
import matplotlib.pyplot as plt
a1 = np.load('pointn2_edge_cases/data1.npy')
a2 = np.load('pointn2_edge_cases/label1.npy')
a3 = np.load('pointn2_edge_cases/pred1.npy')[0]



args  = sys.argv
vis = args[-2]
ind = int(args[-1])
print (ind)
def convert_one_hot_to_label(label):
    return np.argmax(label,axis  = -1)

a4 = convert_one_hot_to_label(a3)

def unscale_points(pcl):
    pcl[:,0] *= 40
    pcl[:,1] *= 30
    return pcl


a5 = unscale_points(a1[ind])
if vis  == "vispcl":
    read_kitti_data.visualize_results(a5,a2[ind])
    read_kitti_data.visualize_results(a5,a4[ind])

# visualize_results



bp = birds_eye_view()
c1  = np.reshape(a2[ind],(60000,1))
pcl = np.hstack((a1[ind],c1))
# pcl = unscale_points(pcl)
p1 = bp.get_birds_eye_view(pcl)
plt.subplot(1,2,1)
plt.title('Ground Label')
plt.imshow(p1)



bp1 = birds_eye_view()
c2  = np.reshape(a4[ind],(60000,1))
pcl2 = np.hstack((a1[ind],c2))
p2 = bp1.get_birds_eye_view(pcl2)
plt.subplot(1,2,2)
plt.title('Vanilla Pointnet++_ Epoch 18')
plt.imshow(p2)
plt.show()
