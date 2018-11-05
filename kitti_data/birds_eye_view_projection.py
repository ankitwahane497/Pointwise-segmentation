from utils import *
import numpy as np
class birds_eye_view:
    def __init__(self, target_height = 800, target_width = 400, x_range = 60,
        y_range = 20,z_range = 10,channels =3):
        self.target_height = target_height
        self.target_width  = target_width
        self.channels      = channels
        self.x_range       = x_range
        self.y_range       = y_range
        self.z_range       = z_range
        self.scale_x       = (self.target_height/self.x_range)
        self.scale_y       = (self.target_width/(2*self.y_range))
        self.projection    = np.zeros((self.target_height,self.target_width,self.channels)
                                      ,dtype = np.int32)



    def get_birds_eye_view(self, pcl, shift_pcl = True, scale_pcl = True, scale_image = True):
        pcl = filter_range_points(pcl,self.x_range, self.y_range,
                                  self.z_range, False)
        if(shift_pcl):
            pcl = shift_points(pcl, shift_x = True, shift_y = True, y_range = self.y_range)
        if(scale_pcl):
            pcl = scale_points(pcl, x_scale = self.scale_x, y_scale = self.scale_y)

        for i in range(len(pcl)):
            # if(pcl[i,-1] < self.channels):
            try:
                self.projection[int(pcl[i,0]),int(pcl[i,1]),int(pcl[i,-1])] = 1
            except:
                print (int(pcl[i,0]),int(pcl[i,1]),int(pcl[i,-1]))
        if scale_image:
            self.projection *= 255
        return self.projection
