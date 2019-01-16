import pdb
import numpy as np
#
#
#
def filter_range_points(pcl, x_range = 100, y_range_min = 30 , y_range_max= 60,
                        z_range = None, use_full_cloud = False):
    pcl[:,0] -= np.min(pcl[:,0])
    pcl[:,1] -= np.min(pcl[:,1])
    if (x_range != None):
        pcl = pcl[pcl[:,0] > x_range]
    if (use_full_cloud == False):
        pcl = pcl[pcl[:,0] >  0]
    # if (y_range_max != None):
    #     pcl = pcl[pcl[:,1] > y_range_min]
    #     pcl = pcl[pcl[:,1] < y_range_max]
    return pcl

def scale_points(pcl, x_scale = None, y_scale = None):
    if (x_scale != None):
        pcl[:,0] = x_scale*pcl[:,0]
    if (y_scale != None):
        pcl[:,1] = y_scale*pcl[:,1]
    return pcl

def shift_points(pcl,shift_x = False, shift_y= False, y_range = 0):
    if (shift_x == True):
        shift_factor = min(pcl[:,0])
        pcl[:,0]     += abs(shift_factor)
    if (shift_y == True):
        # shift_factor = min(pcl[:,1])
        shift_factor = y_range
        # pdb.set_trace()
        pcl[:,1]     += abs(shift_factor)
    return pcl
