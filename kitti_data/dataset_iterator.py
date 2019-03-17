import numpy as np
import sys
import glob
import pdb
from sklearn.model_selection import train_test_split
from read_kitti_data import *
from cloud_clustering import *

basedir = '/home/sanket/MS_Thesis/kitti'

class Kitti_data_iterator:
    def __init__(self, basedir, batch_size = 32 , num_points = 60000):
        self.basedir = basedir
        self.batch_size = batch_size
        self.num_points = num_points
        self.batch_pointer = 0
        self.get_training_files()
        self.split_training_and_testing()
        self.n_batches = int(len(self.train_data)/self.batch_size)
        self.iteration = 0

    def get_training_files(self):
        files   = glob.glob(self.basedir+'/image_2/*')
        self.frames  = [file.split('/')[-1][:-4] for file in files]

    def split_training_and_testing(self):
        self.train_data , self.test_data = train_test_split(self.frames,
                                            test_size = 0.2)

    def get_batch(self):
        batch_data   = np.zeros((self.batch_size,self.num_points, 4))
        batch_labels_instance = np.zeros((self.batch_size,self.num_points,3))
        batch_labels_seg = np.zeros((self.batch_size,self.num_points))
        self.batch_pointer += self.batch_size
        if (self.batch_pointer > (len(self.train_data) - self.batch_size)):
            #shuffle
            self.batch_pointer = 0
            self.iteration += 1
        for i in range(self.batch_size):
            pcl, label= get_instance_vector_frame_and_label(self.train_data[self.batch_pointer + i])
            pcl = np.concatenate((pcl,np.reshape(label,(-1,4))), axis  =-1)
            pcl = fix_samples(pcl, num_samples = self.num_points)
            batch_data[i] = pcl[:,:4]
            batch_labels_instance[i] = pcl[:,4:7]
            batch_labels_seg[i] = pcl[:,7]
        return batch_data,  batch_labels_instance, batch_labels_seg, self.iteration, self.batch_pointer

    def get_batch_with_images(self):
        batch_data   = np.zeros((self.batch_size,self.num_clusters,self.num_cluster_samples, 4))
        batch_labels = np.zeros((self.batch_size,self.num_clusters,self.num_cluster_samples))
        batch_images = []
        # batch_data = []
        # batch_labels = []
        self.batch_pointer += self.batch_size
        if (self.batch_pointer > (len(self.train_data) - self.batch_size)):
            #shuffle
            self.batch_pointer = 0
            self.iteration += 1
        for i in range(self.batch_size):
            pcl, label , img= get_frame_label_and_image(self.train_data[self.batch_pointer + i])
            pcl = np.concatenate((pcl,np.reshape(label,(-1,1))), axis  =-1)
            pcl = fix_samples(pcl, num_samples = self.num_points)
            pcl = get_Gaussian_labels(pcl,self.num_clusters, self.num_points )
            pcl_cluster = get_cluster(pcl,self.num_clusters, self.num_cluster_samples)
            batch_data[i] = pcl_cluster[:,:,:4]
            batch_labels[i] = pcl_cluster[:,:,-1]
            batch_images.append(img)
        return batch_data, batch_labels, batch_images,self.iteration, self.batch_pointer


if __name__=='__main__':
    data = Kitti_data_iterator(basedir,batch_size = 1 , num_points = 10000)
    data.get_training_files()
    a1 , b1 ,c1, d1, e1 = data.get_batch()
    # a1 , b1 ,c1, d ,e = data.get_batch_with_images()
    pdb.set_trace()
    print ('exit log')
