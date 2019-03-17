import numpy as np
import pdb

def get_pcl_instance_labels(pcl,Bb):
    new_pcl = np.zeros((len(pcl),9))
    new_pcl[:,:4] = pcl[:,:4]
    for i in range(len(Bb)):
        b1 = Bb[i]
        x_max , x_min = max(b1[:,0]), min(b1[:,0])
        y_max , y_min = max(b1[:,1]), min(b1[:,1])
        z_max , z_min = max(b1[:,2]), min(b1[:,2])
        coord = np.where(((new_pcl[:,0] > x_min) & (new_pcl[:,0] < x_max))&
                        ((new_pcl[:,1] > y_min) & (new_pcl[:,1] < y_max)))
        new_pcl[coord,-1] = i+1
        new_pcl[coord,-2] = 1
        # new_pcl[coord,3:6] = (x_max+x_min)/2, (y_max+y_min)/2 , (z_max+z_min)/2
        new_pcl[coord,4:7] = np.mean(b1[:,0]), np.mean(b1[:,1]) , 0
        # print ('In loop ',i)
    # pdb.set_trace()
    return new_pcl
