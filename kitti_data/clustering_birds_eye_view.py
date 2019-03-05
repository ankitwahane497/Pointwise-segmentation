from utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
class clustering_birds_eye_view:
    def __init__(self, target_height = 800, target_width = 400, x_range = 40,
        y_range = 15,z_range = 10,channels =3):
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
        self.instance_projection    = np.full((self.target_height,self.target_width,self.channels)
                                      ,254,dtype = np.int32)



    def get_birds_eye_view(self, pcl, shift_pcl = True, scale_pcl = True, scale_image = True, class_dict = None):
        key_dict = {'Car':1, 'Van': 2, 'Truck':3,
                    'Pedestrian':4, 'Person_sitting':5,
                    'Cyclist': 6 , 'Tram' : 7 ,
                    'Misc' : 0 , 'DontCare': 0}
        class_dict = {0:[1,1,1] , 1:[0,1,0], 2:[0,0,1],3: [1,0,0],
                      4:[0.2,0.3,1], 5 : [0.6,0.3,0.1], 6 :[0.8,0.4,0.3] , 7 :[0.5,0.5,0.5],
                      8:[0.6,0.9,0.1],9:[0.3,0.1,0.9], 10:[0.3,0.3,0.3],11 :[0.4,0.3,0.7],
                      12:[0.7,0.3,0.4],13:[0.4,0.8,0.1],14:[0.25,0.55,0.78], 15:[0.89,0.23,0.67]}
        class_instance_dict = {0:(255,255,0) , 1:(255,0,0), 2:(0,255,0),3: (0,0,255),
                      4:(100,0,50), 5 : (232,52,0), 6 :(12,133,23) , 7 : (53,30,210), 
                      8 : (20,55,156) , 9 : (70,70,70) , 10: (220,20,60) , 11: (107,142,35),
                      12 : (0,0,142), 13: (220,220,0) , 14: (190,153,153), 15 :(157,234,50)}
        pcl = filter_range_points(pcl,self.x_range, 2*self.y_range,
                                  self.z_range, False)
        if(shift_pcl):
            pcl = shift_points(pcl, shift_x = True, shift_y = True, y_range = self.y_range)
        if(scale_pcl):
            pcl = scale_points(pcl, x_scale = self.scale_x, y_scale = self.scale_y)
        for i in range(len(pcl)):
            try:
                self.projection[int(pcl[i,0]),int(pcl[i,1])] = class_dict[int(pcl[i,3])]
            except:
                pass
        self.projection = 1 - self.projection #invert
        if scale_image:
            self.projection *= 255

        for i in range(len(pcl)):
            try:
                if(pcl[i,5] != 0):
                    self.instance_projection[int(pcl[i,0]),int(pcl[i,1])] = class_instance_dict[int(pcl[i,5] % 16)]
            except:
                print ('Problem in instance projection')
                pass        
        return self.projection , self.instance_projection
