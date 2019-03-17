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
import copy
from utils import *
from instance import *
left_cam_rgb= 'image_2'
label = 'label_2'
velodyne = 'velodyne'
calib = 'calib'

# basedir ='/home/srgujar/kitti'
#basedir = '/home/srgujar/Data/training'
basedir = '/home/sanket/MS_Thesis/kitti'
import vispy
from vispy.scene import visuals


from birds_eye_view_projection import birds_eye_view
from instance_birds_projection import instance_birds_eye_view

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
  key_dict = {'Car':1, 'Van': 2, 'Truck':3,
             'Pedestrian':4, 'Person_sitting':5,
             'Cyclist': 6 , 'Tram' : 7 ,
             'Misc' : 0 , 'DontCare': 0}
  bb3d = []
  label_bb = []

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
        label_bb.append(key_dict[key])
  return np.array(bb3d), label_bb

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

def get_pcl_class_label(pcl,Bb, label_bb):
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
        pcl[coord,-1] = label_bb[i]
    return pcl



def cart_to_hom(pcl):
    ##convert cartesian to homogenous
    np_ = pcl.shape[0]
    pcl = np.hstack((pcl,np.ones((np_,1))))
    return pcl

def project_velo_to_rect(pcl,calib):
    pcl = cart_to_hom(pcl[:,:3])
    pcl = np.dot(pcl,np.transpose(calib['Tr_velo_to_cam'].reshape(3,4)))
    # pcl_rect = np.transpose(np.dot(calib['R0_rect'].reshape(3,3),np.transpose(pcl)))
    pcl_rect = np.dot(pcl,calib['R0_rect'].reshape(3,3))
    # img = pcl_rect
    pcl_rect = cart_to_hom(pcl_rect)
    img      = np.dot(pcl_rect, np.transpose(calib['P2'].reshape(3,4)))
    # img      = np.dot(pcl, np.transpose(calib['P2'].reshape(3,4)))

    img[:,0]/= img[:,2]
    img[:,1]/= img[:,2]
    # visualize(np.hstack((img[:,0:2],np.zeros((len(img),1)))))
    return img[:,0:2]

def pcl_to_rect(pcl,calib):
    pcl = cart_to_hom(pcl[:,:3])
    pcl = np.dot(pcl,np.transpose(calib['Tr_velo_to_cam'].reshape(3,4)))
    # pcl_rect = np.transpose(np.dot(calib['R0_rect'].reshape(3,3),np.transpose(pcl)))
    pcl_rect = np.dot(pcl,calib['R0_rect'].reshape(3,3))
    return pcl_rect




def get_lidar_in_fov(pcl,calib,image,clip_dist = 2.0, return_ind = False):
    pts_2d = project_velo_to_rect(pcl,calib)
    img_h , img_w = image.shape[:2]
    Fov_in  = (pts_2d[:,0] > 0) & (pts_2d[:,0] < img_w) & \
              (pts_2d[:,1] > 0) & (pts_2d[:,1] < img_h)
    Fov_in  = Fov_in & (pcl[:,0] > clip_dist)
    pcl_fov = pcl[Fov_in,:]
    if return_ind == True:
        return pcl_fov, pts_2d, Fov_in
    else:
        return pcl_fov


def get_lidar_projection(pcl,calib, left_cam_image, on_image = True):
    pcl_fov, pts_2d, fov_ind = get_lidar_in_fov(pcl,calib,
                                np.array(left_cam_image),
                                return_ind = True)
    left_cam_image = np.array(left_cam_image)
    img_h , img_w = left_cam_image.shape[:2]
    img_pts_fov = pts_2d[fov_ind,:]

    proj_img = np.zeros((img_w + 1, img_h + 1,3))
    for i in range(img_pts_fov.shape[0]):
        try:
            proj_img[int(np.round(img_pts_fov[i,0])),int(np.round(img_pts_fov[i,1])),0] = 1
        except:
            # print (img_labels[i])
            pass

    if on_image:
        pcl_fov_rect = pcl_to_rect(pcl_fov,calib)

        cmap = plt.cm.get_cmap('hsv',256)
        cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

        for i in range(img_pts_fov.shape[0]):
            depth = pcl_fov_rect[i,2]
            color = cmap[int(640./depth),:]
            cv2.circle(left_cam_image, (int(np.round(img_pts_fov[i,0])),
                                        int(np.round(img_pts_fov[i,1]))),
                                        2, color = tuple(color) , thickness = -1)
        return cv2.flip(np.rot90(proj_img,3),1) , left_cam_image
    else:
        return cv2.flip(np.rot90(proj_img,3),1)

def get_lidar_projection_with_labels(pcl,calib, left_cam_image, on_image = True):
    labels = copy.deepcopy(pcl[:,-1])
    pcl    = pcl[:,:3]
    pcl_fov, pts_2d, fov_ind = get_lidar_in_fov(pcl,calib,
                                np.array(left_cam_image),
                                return_ind = True)
    left_cam_image = np.array(left_cam_image)
    img_h , img_w = left_cam_image.shape[:2]
    img_pts_fov = pts_2d[fov_ind,:]
    img_labels  = labels[fov_ind]

    proj_img = np.zeros((img_w + 1, img_h + 1,3))
    for i in range(img_pts_fov.shape[0]):
        try:
            proj_img[int(np.round(img_pts_fov[i,0])),int(np.round(img_pts_fov[i,1])),int(img_labels[i])] = 1
        except:
            print (img_labels[i])

    if on_image:
        pcl_fov_rect = pcl_to_rect(pcl_fov,calib)

        cmap = plt.cm.get_cmap('hsv',256)
        cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255
        red = [255,0,255]

        for i in range(img_pts_fov.shape[0]):
            depth = pcl_fov_rect[i,2]
            color = cmap[int(640./depth),:]
            if img_labels[i] == 0:
                cv2.circle(left_cam_image, (int(np.round(img_pts_fov[i,0])),
                                            int(np.round(img_pts_fov[i,1]))),
                                            2, color = tuple(color) , thickness = -1)
            else:
                cv2.circle(left_cam_image, (int(np.round(img_pts_fov[i,0])),
                                            int(np.round(img_pts_fov[i,1]))),
                                            2, color = red , thickness = -1)

        return cv2.flip(np.rot90(proj_img,3),1) , left_cam_image
    else:
        return cv2.flip(np.rot90(proj_img,3),1)




def visualize(pcl,label = None):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    # pcl = convert_pcl(pcl,label)
    if (label != None):
        pcl = convert_pcl(pcl,label)
    if (label != None):
        pcl = add_points(pcl,label)
    # pcl = get_road(pcl)
    if (label != None):
        scatter.set_data(pcl[:,:3],edge_color = color_b(pcl[:,-1]), size = 2)
    else:
        scatter.set_data(pcl[:,:3], size = 7)

    # scatter.set_data(, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    view.add(scatter)

    view.camera = 'turntable'

    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()
    # pdb.set_trace()

def color_labels(labels):
    colored_labels  = np.zeros((len(labels),3))
    for i in range(len(labels)):
        if labels[i] == 0:
            pass
        else:
            colored_labels[i][0] = 1
    return colored_labels

def visualize_results(pcl,label = None):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    # pcl = convert_pcl(pcl,label)
    if (label.any() != None):
        scatter.set_data(pcl[:,:3],edge_color = color_labels(label), size = 2)
    else:
        scatter.set_data(pcl[:,:3], size = 7)

    # scatter.set_data(, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    view.add(scatter)

    view.camera = 'turntable'

    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()
    # pdb.set_trace()



def get_frame_and_label(frame):
      left_cam, velo, label_data, calib_data = loadKittiFiles(frame)
      # pdb.set_trace()
      velo = np.hstack((velo,np.zeros((len(velo),1))))
      bb3d, label_bb = get_3D_BoundingBox(label_data, calib_data)
      pcl = get_pcl_class_label(velo, bb3d, label_bb)
      pcl = filter_range_points(pcl, x_range = 40, y_range = 15)
      # pdb.set_trace()
      pcl = shift_points(pcl, shift_y = True, y_range = 15)
      pcl = scale_points(pcl,y_scale = (1/30), x_scale = (1/40))
      # visualize_results(pcl)
      return pcl[:,:4], pcl[:,-1]

def get_frame_label_and_image(frame):
      left_cam, velo, label_data, calib_data = loadKittiFiles(frame)
      # pdb.set_trace()
      velo = np.hstack((velo,np.zeros((len(velo),1))))
      bb3d, label_bb = get_3D_BoundingBox(label_data, calib_data)
      pcl = get_pcl_class_label(velo, bb3d, label_bb)
      pcl = filter_range_points(pcl, x_range = 40, y_range = 15)
      # pdb.set_trace()
      pcl = shift_points(pcl, shift_y = True, y_range = 15)
      pcl = scale_points(pcl,y_scale = (1/30), x_scale = (1/40))
      # visualize_results(pcl)
      return pcl[:,:4], pcl[:,-1], left_cam


def get_instance_vector_frame_and_label(frame):
    left_cam, velo, label_data, calib_data = loadKittiFiles(frame)
    bb3d, label_bb = get_3D_BoundingBox(label_data, calib_data)
    pcl      = get_pcl_class_label(velo, bb3d, label_bb)
    pcl = get_pcl_instance_labels(pcl, bb3d)
    pcl = filter_range_points(pcl, x_range = 40, y_range = 15)
    pcl = shift_points_with_instance(pcl, shift_y = True, y_range = 15)
    pcl = scale_points_with_instance(pcl,y_scale = (1/30), x_scale = (1/40))
    # visualize_instance(pcl  , left_cam)
    return pcl[:,:4], pcl[:,4:-1]

def visualize_instance(it_label, left_cam):
    pr2  = instance_birds_eye_view()
    sem_img, inst_img = pr2.get_birds_eye_view(it_label,shift_pcl = False)
    sem_img = convert_image_plot(sem_img)
    inst_img= convert_image_plot(inst_img)
    plt.subplot(1,3,1)
    plt.imshow(left_cam)
    plt.title('Camera Image')
    plt.subplot(1,3,2)
    plt.imshow(sem_img)
    plt.title('Semantic')
    plt.subplot(1,3,3)
    plt.imshow(inst_img)
    plt.title('Instance Segmentation')
    plt.tight_layout()
    plt.show()



def main_frame (frame='000008'):
  """
  Completes the plots
  """
  # p1,p2 = get_instance_vector_frame_and_label(frame)
  left_cam, velo, label_data, calib_data = loadKittiFiles(frame)
  bb3d, label_bb = get_3D_BoundingBox(label_data, calib_data)
  # proj, cam_img = get_lidar_projection(velo,calib_data, left_cam)
  pcl      = get_pcl_class_label(velo, bb3d, label_bb)
  it_label = get_pcl_instance_labels(pcl, bb3d)
  pr2  = instance_birds_eye_view()
  sem_img, inst_img = pr2.get_birds_eye_view(it_label)
  sem_img = convert_image_plot(sem_img)
  inst_img= convert_image_plot(inst_img)

  plt.subplot(1,3,1)
  plt.imshow(left_cam)
  plt.title('Camera Image')
  plt.subplot(1,3,2)
  plt.imshow(sem_img)
  plt.title('Semantic')
  plt.subplot(1,3,3)
  plt.imshow(inst_img)
  plt.title('Instance Segmentation')
  plt.tight_layout()
  plt.show()

  # plt.subplot(2,2,1)
  # plt.imshow(left_cam)
  # plt.title('Camera Image')
  # plt.subplot(2,2,2)
  # plt.imshow(proj)
  # plt.title('LiDAR Front view projection with labels')
  # plt.subplot(2,2,3)
  # plt.imshow(cam_img)
  # plt.title('LiDAR Front view projection with labels on Image')
  # plt.subplot(2,2,4)
  # plt.imshow(img)
  # plt.title('Birds eye view with labels')
  # plt.show()
  #


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
  main_frame(frame=FLAGS.frame)
