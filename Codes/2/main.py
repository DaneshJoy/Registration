import numpy as np
from utils import bounding_box
from visualization import VisualizePoints


# %% Reading (x y z) 3D points from text file
print('├──■ Loading XYZ Points ...')
fixed = np.loadtxt('../../Data/SurfaceRegOutput2/OuterLayerPoints.txt')
moving = np.loadtxt('../../Data/SurfaceRegOutput2/surfacePoints_Out1.txt')
landmarks = np.loadtxt('../../Data/SurfaceRegOutput2/landmarks.txt')
_n = len(fixed)
fixed = fixed[300000:]
print('│ └──■ Loaded: \n' +
      f'│\t●{_n:─>8} total fixed points \n' +
      f'│\t●{len(fixed):─>8} selected fixed points \n' +
      f'│\t●{len(moving):─>8} moving points')

# %% Crop fixed points by bounding box
min_pts = [np.min(moving[:, [0]]),
           np.min(moving[:, [1]]),
           np.min(moving[:, [2]])]
max_pts = [np.max(moving[:, [0]]),
           np.max(moving[:, [1]]),
           np.max(moving[:, [2]])]

outer_bbox = bounding_box(fixed,
                          min_x=min_pts[0]-5, max_x=max_pts[0]+5,
                          min_y=min_pts[1]-5, max_y=max_pts[1],
                          min_z=min_pts[2]-5, max_z=max_pts[2]+5)
fixed_cropped = fixed[outer_bbox]
print(f'│\t●{len(fixed_cropped):─>8} fixed points cropped by outer bbox')

inner_bbox = bounding_box(fixed_cropped,
                          min_x=min_pts[0]+5, max_x=max_pts[0]-5,
                          min_y=min_pts[1]+5, max_y=max_pts[1],
                          min_z=min_pts[2]+5, max_z=max_pts[2]-5)
fixed_cropped = fixed_cropped[~inner_bbox]
print(f'│\t●{len(fixed_cropped):─>8} fixed points cropped by inner bbox')

# %% Visualization
vis = VisualizePoints()

vis.set_data(fixed_cropped, moving, landmarks)
vis.show_side_by_side()
# vis.show_bbox(moving, ax=vis.ax_single)