import numpy as np
import vispy
from plyfile import PlyData, PlyElement
import pdb
import vispy.scene
from vispy.scene import visuals
from birds_eye_view_projection import birds_eye_view
import matplotlib.pyplot as plt
# from sklearn.preprocessing import normalize

plydata = PlyData.read('00500.ply')

datalen = len(plydata['vertex']['x'])
pcl = np.zeros((datalen,6))
for i in range(datalen):
	data_ = plydata['vertex'].data[i]
	for j in range(6):
		pcl[i][j] = data_[j]
# pcl = plydata['vertex'].data[0]
# pdb.set_trace()

def labels_to_cityscapes_palette(pcl):
	# classes = { [220, 20, 60] : 2,  [0, 0, 255]   : 1  }
	# pdb.set_trace()
	for i in range(len(pcl)):
		try:
			if (pcl[i,4] ==  20):
				pcl[i,-1]  = 2
			elif (pcl[i,-1] ==  255):
				pcl[i,-1]  = 1
			else:
				pcl[i,-1] = 0
		except:
			pcl[i,-1]  = 0
	return pcl


bV = birds_eye_view(channels=3)
pcl = labels_to_cityscapes_palette(pcl)
pro = bV.get_birds_eye_view(pcl)
plt.imshow(pro)
plt.show()
# pdb.set_trace()

# def normalize(data):
# 	# data_ = normalize(data)
# 	data_ = data/255
# 	# pdb.set_trace()
# 	return data_
#
#

#
# Make a canvas and add simple view
#
# canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
# view = canvas.central_widget.add_view()
#

# color_data = normalize(pcl[:,3:])


# # create scatter object and fill in the data
# scatter = visuals.Markers()
# scatter.set_data(pcl[:,:3], edge_color = color_data, size = 3)
# view.add(scatter)
#
# view.camera = 'turntable'
#
# axis = visuals.XYZAxis(parent=view.scene)
# vispy.app.run()
