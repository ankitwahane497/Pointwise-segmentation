import pdb
import numpy as np
#
#
#
def filter_range_points(pcl, x_range = None, y_range = None ,
                        z_range = None, use_full_cloud = False):
    if (x_range != None):
        pcl = pcl[pcl[:,0] < x_range]
    if (use_full_cloud == False):
        pcl = pcl[pcl[:,0] >  0]
    if (y_range != None):
        pcl = pcl[pcl[:,1] < y_range]
        pcl = pcl[pcl[:,1] > -y_range]
    # if (z_range != None):
    #     pcl = pcl[pcl[:,2] < z_range]
    #     pcl = pcl[pcl[:,2] > -z_range]
    return pcl


def scale_points(pcl, x_scale = None, y_scale = None):
    if (x_scale != None):
        pcl[:,0] = x_scale*pcl[:,0]
    if (y_scale != None):
        pcl[:,1] = y_scale*pcl[:,1]
    return pcl

def scale_points_with_instance(pcl, x_scale = None, y_scale = None):
    if (x_scale != None):
        pcl[:,0] = x_scale*pcl[:,0]
        pcl[:,4] = x_scale*pcl[:,4]
    if (y_scale != None):
        pcl[:,1] = y_scale*pcl[:,1]
        pcl[:,5] = y_scale*pcl[:,5]
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


def shift_points_with_instance(pcl,shift_x = False, shift_y= False, y_range = 0):
    if (shift_x == True):
        shift_factor = min(pcl[:,0])
        pcl[:,0]     += abs(shift_factor)
        pcl[:,4]     += abs(shift_factor)
    if (shift_y == True):
        shift_factor = y_range
        pcl[:,1]     += abs(shift_factor)
        pcl[:,5]     += abs(shift_factor)
    return pcl


def fix_samples(pcl,num_samples = 2000):
    pcl_shape = pcl.shape[-1]
    new_pcl = np.zeros((num_samples,pcl_shape))
    if (len(pcl) >= num_samples):
        indx = np.random.choice(len(pcl), num_samples, replace=False)
        new_pcl = pcl[indx][:,:pcl_shape]
    else:
        new_pcl[:len(pcl)] = pcl[:,:pcl_shape]
    return new_pcl
