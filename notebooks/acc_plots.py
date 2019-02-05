import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
# data to plot
n_groups = 3
means_frank = (95.15,93.63,93.3)
means_guido = (56.07,76.6,82.86)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Accuracy')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Target Class Accuracy')

# plt.xlabel('')
plt.ylabel('Pointwise Accuracy')
plt.title('Comparison of Pointwise methods on Kitti Dataset')
plt.xticks(index + 0.5* bar_width, ('Pointnet++', 'Edge Conv', 'Proposed Method'))
# Shrink current axis by 20%

ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
# plt.grid()
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Comparision.png', bbox_inches='tight', dpi=300)
plt.show()
