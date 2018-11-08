import numpy as np
import sys
import glob
import pdb
from sklearn.model_selection import train_test_split
basedir = '/media/sanket/My Passport/Sanket/Kitti/training'
# sys.path.append('/home/sanket/MS_Thesis/Pointwise-segmentation/kitti_data')
from read_kitti_data import *



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
        # p1,p2 = self.get_batch()
        # pdb.set_trace()

    def get_training_files(self):
        files   = glob.glob(self.basedir+'/image_2/*')
        self.frames  = [file.split('/')[-1][:-4] for file in files]

    def split_training_and_testing(self):
        self.train_data , self.test_data = train_test_split(self.frames,
                                            test_size = 0.2)

    def get_batch(self):
        batch_data   = np.zeros((self.batch_size,self.num_points, 3))
        batch_labels = np.zeros((self.batch_size,self.num_points))
        # batch_data = []
        # batch_labels = []
        self.batch_pointer += self.batch_size
        if (self.batch_pointer > (len(self.train_data) - self.batch_size)):
            #shuffle
            self.batch_pointer = 0
            self.iteration += 1
        for i in range(self.batch_size):
            pcl, label = get_frame_and_label(self.train_data[self.batch_pointer + i])
            if (len(pcl) >= self.num_points):
                batch_data[i] = pcl[:self.num_points]
                batch_labels[i] = label[:self.num_points]
            else:
                batch_data[i,:len(pcl)] = pcl
                batch_data[len(pcl):]   = pcl[-1]
                batch_labels[i,:len(pcl)] = label
                batch_labels[len(pcl):]   = label[-1]
        return batch_data, batch_labels, self.iteration, self.batch_pointer

if __name__=='__main__':
    data = Kitti_data_iterator(basedir)
    data.get_training_files()
