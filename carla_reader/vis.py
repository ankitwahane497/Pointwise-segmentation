import warnings
warnings.filterwarnings("ignore")
import numpy as np
import vispy
from plyfile import PlyData, PlyElement
import pdb
import vispy.scene
from vispy.scene import visuals
# from sklearn.preprocessing import normalize

plydata = PlyData.read('rain_town1/00200.ply')

datalen = len(plydata['vertex']['x'])
pcl = np.zeros((datalen,6))
for i in range(datalen):
	data_ = plydata['vertex'].data[i]
	for j in range(6):
		pcl[i][j] = data_[j]
# pcl = plydata['vertex'].data[0]
# pdb.set_trace()


def normalize(data):
	# data_ = normalize(data)
	data_ = data/255
	# pdb.set_trace()
	return data_

#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


color_data = normalize(pcl[:,3:])


# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pcl[:,:3], edge_color = color_data, size = 3)
view.add(scatter)

view.camera = 'turntable'

axis = visuals.XYZAxis(parent=view.scene)
vispy.app.run()
