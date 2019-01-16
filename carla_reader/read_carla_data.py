import warnings
warnings.filterwarnings("ignore")
import numpy as np
import vispy
from plyfile import PlyData, PlyElement
import pdb
from utils import *


# carla_classes = {
#     [0, 0, 0] :,         # None
#     1: [70, 70, 70],      # Buildings
#     2: [190, 153, 153],   # Fences
#     3: [72, 0, 90],       # Other
# }


carla_label = { ( 0, 0, 0) : 0,
                ( 70, 70, 70):1,
                (190, 153, 153):2,
                (72, 0, 90):3,
                (220, 20, 60):4,
                (153, 153, 153):5,
                (157, 234, 50):6,
                (128, 64, 128):7,
                (244, 35, 232):8,
                (107, 142, 35):9,
                ( 0, 0, 255):10,
                (102, 102, 156):11,
                (220, 220, 0):12 }
# for key in carla_label:
#     print (key)


def convert_rgb_to_label(rbg_label):
    try:
        label = carla_label[tuple(rbg_label)]
    except:
        label = 0
    return int(label)

def read_frame_with_class_label(frame):
    plydata = PlyData.read(frame)
    datalen = len(plydata['vertex']['x'])
    pcl = np.zeros((datalen,4))
    for i in range(datalen):
        data_ = plydata['vertex'].data[i]
        rgb_label = []
        for j in range(3):
            pcl[i][j] = data_[j]
        for j in range(3,6):
            rgb_label.append(data_[j])
        pcl[i][3] = convert_rgb_to_label(rgb_label)
    return pcl

def read_frame_with_rgb_labels(frame):
    plydata = PlyData.read(frame)
    datalen = len(plydata['vertex']['x'])
    pcl = np.zeros((datalen,6))
    for i in range(datalen):
    	data_ = plydata['vertex'].data[i]
    	for j in range(6):
    		pcl[i][j] = data_[j]
    return pcl

def get_frame_and_label(frame):
    try:
        pcl = read_frame_with_class_label(frame)
        pcl = filter_range_points(pcl)
        pcl = scale_points(pcl,y_scale = (1/max(pcl[:,1])), x_scale = (1/max(pcl[:,0])))
    except:
        # pdb.set_trace()
        return np.zeros((10000,3)),np.zeros((10000)), -1
    return pcl[:,:3], pcl[:,-1], 1


if __name__ =='__main__':
    frame ='_out/00100.ply'
    frame = read_frame_with_class_label(frame)
    # frame  = read_frame_with_rgb_labels(frame)
    pdb.set_trace()
    print ('*****')
