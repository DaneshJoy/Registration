from functools import partial

import numpy as np
from pycpd import RigidRegistration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from visualization import VisualizePoints, cuboid_data
from point_utils import bounding_box


# %% Reading (x y z) 3D points from text file
print('├──■ Loading XYZ Points ...')
target_pts = np.loadtxt('../../Data/SurfaceRegOutput2/OuterLayerPoints.txt')
source_pts = np.loadtxt('../../Data/SurfaceRegOutput2/surfacePoints_Out1.txt')
landmark_pts = np.loadtxt('../../Data/SurfaceRegOutput2/landmarks.txt')
target_pts = target_pts[300000:]
print('│ └──■ Loaded: \n' +
      f'│\t● {len(target_pts):─>8} target points \n' +
      f'│\t● {len(source_pts):─>8} source points')

# target_pts = target_pts[:, [0, 1, 2]]
min_pts = [np.min(source_pts[:, [0]]-10),
           np.min(source_pts[:, [1]]-10),
           np.min(source_pts[:, [2]]-10)]
max_pts = [np.max(source_pts[:, [0]]+10),
           np.max(source_pts[:, [1]]+10),
           np.max(source_pts[:, [2]]+10)]

inside_box = bounding_box(target_pts,
                          min_x=min_pts[0], max_x=max_pts[0],
                          min_y=min_pts[1], max_y=max_pts[1],
                          min_z=min_pts[2], max_z=max_pts[2])



size = [(max_pts[0]-min_pts[0]),
        (max_pts[1]-min_pts[1]),
        (max_pts[2]-min_pts[2])]
center = [size[0]//2, size[1]//2, size[2]//2]
X, Y, Z = cuboid_data(center, (size[0], size[1], size[2]))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(X, Y, Z, color='r', rstride=1, cstride=1, alpha=0.9)

target_pts = target_pts[inside_box]

in_box = bounding_box(target_pts,
                      min_x=min_pts[0]+10, max_x=max_pts[0]-10,
                      min_y=min_pts[1]+10, max_y=max_pts[1]-10,
                      min_z=min_pts[2]+10, max_z=max_pts[2]-10)
target_pts = target_pts[~in_box]
# %% Visualize Target/Source Points

print('├──■ Visualizing Points ...')
vis = VisualizePoints()
vis.show_data(target_pts, source_pts, landmark_pts)

# %% Register Points

print('└──■ Start Registration ...')
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
fig.tight_layout()
callback = partial(vis.visualize_reg, ax=axes)

reg = RigidRegistration(**{'X': target_pts, 'Y': source_pts})
reg.register(callback)
plt.show()

RR = reg.R
tt = reg.t
reg_mat_1 = np.array([[RR[0][0], RR[0][1], RR[0][2], tt[0]],
                      [RR[1][0], RR[1][1], RR[1][2], tt[1]],
                      [RR[2][0], RR[2][1], RR[2][2], tt[2]],
                      [0,         0,      0,          1]])

print(reg_mat_1)
source_pts_registered = reg.s * np.dot(source_pts, RR) + tt

vis.show_data(target_pts, source_pts_registered)



