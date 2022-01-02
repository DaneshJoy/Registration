from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import RigidRegistration
from visualization import VisualizePoints


target_pts = np.loadtxt('../../Tests/test1/fixed_cropped_inner.txt')
source_pts = np.loadtxt('../../Tests/test1/surface_sampled.txt')


vis = VisualizePoints()
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