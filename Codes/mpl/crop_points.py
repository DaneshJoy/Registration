import numpy as np
from utils import bounding_box
from visualization import VisualizePoints


# %% Reading (x y z) 3D points from text file
print('├──■ Loading XYZ Points ...')
fixed = np.loadtxt('../../Tests/test1/data/SurfaceRegOutput2/OuterLayerPoints.txt')
moving = np.loadtxt('../../Tests/test1/data/SurfaceRegOutput2/surfacePoints_Out1.txt')
landmarks = np.loadtxt('../../Tests/test1/data/SurfaceRegOutput2/landmarks.txt')
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
                          min_y=min_pts[1]-5, max_y=max_pts[1]-5,
                          min_z=min_pts[2]+5, max_z=max_pts[2]+5)
fixed_cropped_outer = fixed[outer_bbox]
print(f'│\t●{len(fixed_cropped_outer):─>8} fixed points cropped by outer bbox')

inner_bbox = bounding_box(fixed_cropped_outer,
                          min_x=min_pts[0]+5, max_x=max_pts[0]-5,
                          min_y=min_pts[1]+5, max_y=max_pts[1],
                          min_z=min_pts[2]+5, max_z=max_pts[2]-5)
fixed_cropped_inner = fixed_cropped_outer[~inner_bbox]
print(f'│\t●{len(fixed_cropped_inner):─>8} fixed points cropped by inner bbox')

try:
      np.savetxt('../../Tests/test1/fixed_cropped_outer.txt',
            fixed_cropped_outer, delimiter='\t', fmt='%.6f')

      np.savetxt('../../Tests/test1/fixed_cropped_inner.txt',
            fixed_cropped_inner, delimiter='\t', fmt='%.6f')
except Exception as e:
      print(f'--X Saving Results Failed:\n\t{e}')

# %% Visualization
vis = VisualizePoints()

# vis.set_data(fixed, moving, landmarks)
# vis.show_side_by_side(show_outer_bbox=False, show_inner_bbox=False)
vis.set_data(fixed_cropped_outer, moving, landmarks)
vis.show_side_by_side(show_outer_bbox=True, show_inner_bbox=False)
vis.set_data(fixed_cropped_inner, moving, landmarks)
vis.show_side_by_side(show_outer_bbox=True, show_inner_bbox=True)
# vis.show_bbox(moving, ax=vis.ax_single)