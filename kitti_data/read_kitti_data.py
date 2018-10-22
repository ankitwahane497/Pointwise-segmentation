#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pdb
import cv2
import os.path
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image
from math import sin, cos
import argparse
import numpy as np

left_cam_rgb= 'image_2'
label = 'label_2'
velodyne = 'velodyne'
calib = 'calib'
basedir = '/media/sanket/My Passport/Sanket/Kitti/training'


def loadKittiFiles (frame) :
  '''
  Load KITTI image (.png), calibration (.txt), velodyne (.bin), and label (.txt),  files
  corresponding to a shot.

  Args:
    frame :  name of the shot , which will be appended to externsions to load
                the appropriate file.
  '''
  # load image file
  fn = basedir+ left_cam_rgb + frame+'.png'
  fn = os.path.join(basedir, left_cam_rgb, frame+'.png')
  left_cam = Image.open(fn).convert ('RGB')

  # load velodyne file
  fn = basedir+ velodyne + frame+'.bin'
  fn = os.path.join(basedir, velodyne, frame+'.bin')
  velo = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)

  # load calibration file
  fn = basedir+ calib + frame+'.txt'
  fn = os.path.join(basedir, calib, frame+'.txt')
  calib_data = {}
  with open (fn, 'r') as f :
    for line in f.readlines():
      if ':' in line :
        key, value = line.split(':', 1)
        calib_data[key] = np.array([float(x) for x in value.split()])

  # load label file
  fn = basedir+ label + frame+'.txt'
  fn = os.path.join(basedir, label, frame+'.txt')
  label_data = {}
  with open (fn, 'r') as f :
    for line in f.readlines():
      if len(line) > 3:
        key, value = line.split(' ', 1)
        #print ('key', key, 'value', value)
        if key in label_data.keys() :
          label_data[key].append([float(x) for x in value.split()] )
        else:
          label_data[key] =[[float(x) for x in value.split()]]

  for key in label_data.keys():
    label_data[key] = np.array( label_data[key])

  return left_cam, velo, label_data, calib_data



def computeBox3D(label, R0,T2):
  '''
  takes an object label and a projection matrix (P) and projects the 3D
  bounding box into the image plane.

  (Adapted from devkit_object/matlab/computeBox3D.m)
  Args:
    label -  object label list or array
  '''
  w = label[7]
  h = label[8]
  l = label[9]
  x = label[10]
  y = label[11]
  z = label[12]
  ry = label[13]

  # compute rotational matrix around yaw axis
  R = np.array([ [+cos(ry), 0, +sin(ry)],
                 [0, 1,               0],
                 [-sin(ry), 0, +cos(ry)] ] )

  # 3D bounding box corners

  x_corners = [0, l, l, l, l, 0, 0, 0] # -l/2
  y_corners = [0, 0, h, h, 0, 0, h, h] # -h
  z_corners = [0, 0, 0, w, w, w, w, 0] # --w/2

  x_corners += -l/2
  y_corners += -h
  z_corners += -w/2

  # bounding box in object co-ordinate
  corners_3D = np.dot(R,np.vstack([x_corners, y_corners, z_corners]))
  corners_3D[0,:] = corners_3D[0,:] + x
  corners_3D[1,:] = corners_3D[1,:] + y
  corners_3D[2,:] = corners_3D[2,:] + z
  corners_3D = project_rect_to_velo(corners_3D,R0,T2)
  return corners_3D

def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Output: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
    return pts_3d_hom

def project_rect_to_velo(coor,R0,T2):
    pts_3d_ref = np.transpose(np.dot(np.linalg.inv(R0), (coor)))
    pts_3d_ref = cart2hom(pts_3d_ref)
    velo_cor = np.dot(pts_3d_ref, np.transpose(T2))
    return velo_cor


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
    [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def get_2D_BoundingBox(labeld):
  """
  Return 2D bounding boxes
  """
  bb2d = []
  for key in labeld.keys ():
    for o in range( labeld[key].shape[0]):
      left   = labeld[key][o][3]
      bottom = labeld[key][o][4]
      width  = labeld[key][o][5]- labeld[key][o][3]
      height = labeld[key][o][6]- labeld[key][o][4]

      xc = (labeld[key][o][5]+labeld[key][o][3])/2
      yc = (labeld[key][o][6]+labeld[key][o][4])/2
      bb2d.append([xc,yc])
  return np.array(bb2d)



def get_3D_BoundingBox(labeld, calibd):
  """
  Return 3D bounding boxes
  """
  T1 = calibd['Tr_velo_to_cam'].reshape(3,4)
  T2 = inverse_rigid_trans(T1)

  bb3d = []

  for key in labeld.keys ():
    for o in range( labeld[key].shape[0]):
      w3d = labeld[key][o][7]
      h3d = labeld[key][o][8]
      l3d = labeld[key][o][9]
      x3d = labeld[key][o][10]
      y3d = labeld[key][o][11]
      z3d = labeld[key][o][12]
      yaw3d = labeld[key][o][13]

      if key != 'DontCare' :
        corners_3D = computeBox3D(labeld[key][o],calibd['R0_rect'].reshape(3,3),T2)
        bb3d.append(corners_3D)
  return np.array(bb3d)

def add_Bbox_points_to_pcl(pcl1,Bb):
    """
    Adds 3D bounding boxes points to the pcl for visulization
    """
    b1 = Bb[0]
    pcl = np.zeros((len(pcl1)+8*len(Bb),4))
    pcl[:-8*len(Bb),:] = pcl1
    for i in range(len(Bb)):
        b1 = Bb[i]
        for j in range(8):
            pcl[-i*8-8+j] = [b1[j,0],b1[j,1],b1[j,2],5]
    return pcl

def get_pcl_class_label(pcl,Bb):
    """
    Label all the points inside a class for a 3D Bbox
    """
    pcl[:,-1] = 0
    for i in range(len(Bb)):
        b1 = Bb[i]
        x_max , x_min = max(b1[:,0]), min(b1[:,0])
        y_max , y_min = max(b1[:,1]), min(b1[:,1])
        z_max , z_min = max(b1[:,2]), min(b1[:,2])
        coord = np.where(((pcl[:,0] > x_min) & (pcl[:,0] < x_max))&
                        ((pcl[:,1] > y_min) & (pcl[:,1] < y_max)))
        pcl[coord,-1] = 1
    return pcl

def get_pcl_instance_labels(pcl,Bb):
    pcl[:,-1] = 0
    new_pcl = np.zeros((len(pcl),7))
    new_pcl[:,:3] = pcl[:,:3]
    for i in range(len(Bb)):
        b1 = Bb[i]
        x_max , x_min = max(b1[:,0]), min(b1[:,0])
        y_max , y_min = max(b1[:,1]), min(b1[:,1])
        z_max , z_min = max(b1[:,2]), min(b1[:,2])
        coord = np.where(((new_pcl[:,0] > x_min) & (new_pcl[:,0] < x_max))&
                        ((new_pcl[:,1] > y_min) & (new_pcl[:,1] < y_max)))
        new_pcl[coord,-1] = i+1
        # new_pcl[coord,3:6] = (x_max+x_min)/2, (y_max+y_min)/2 , (z_max+z_min)/2
        new_pcl[coord,3:6] = np.mean(b1[:,0]), np.mean(b1[:,1]) , 0
    # pdb.set_trace()
    return new_pcl


def visualize(pcl,label):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    # pcl = convert_pcl(pcl,label)
    pcl = filter_points(pcl)
    pcl = convert_pcl(pcl,label)
    pcl = add_points(pcl,label)
    # pcl = get_road(pcl)

    scatter.set_data(pcl[:,:3],edge_color = color_b(pcl[:,-1]), size = 2)
    # scatter.set_data(, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    view.add(scatter)

    view.camera = 'turntable'

    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()
    # pdb.set_trace()
def main (frame='000008'):
  """
  Completes the plots
  """
  left_cam, velo, label_data, calib_data = loadKittiFiles(frame)
  bb3d = get_3D_BoundingBox(label_data, calib_data)




def convert_image_plot(img):
    img = cv2.flip(img, 0)
    img = cv2.flip(img, 1)
    return img

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--frame', type=str,
                      default='000008',
                      help='frame name without extension')
  FLAGS, unparsed = parser.parse_known_args()
  #print ('FLAGS', FLAGS)
  main(frame=FLAGS.frame)
