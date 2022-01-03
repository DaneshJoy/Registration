import numpy as np
from visualization import VisualizePoints
import random as rnd
from tqdm import tqdm


cropped_points_outer = np.loadtxt('../../Tests/test1/fixed_cropped_outer.txt')
cropped_points_inner = np.loadtxt('../../Tests/test1/fixed_cropped_inner.txt')

# %% Random Sample Surface Points
for i in tqdm(range(100)):
    is_ok = False
    region_points = 6
    surface_points_region1 = []
    surface_points_region2 = []
    surface_points_region3 = []
    while not is_ok:
        point_ = rnd.choice(cropped_points_inner)
        if 245<point_[0]<250 and 223<point_[1]<233 and 202<point_[2]<208  and len(surface_points_region1) < region_points:
            surface_points_region1.append(point_)
        elif 210<point_[0]<225 and 227<point_[1]<233 and 200<point_[2]<208  and len(surface_points_region2) < region_points:
            surface_points_region2.append(point_)
        elif 225<point_[0]<235 and 206<point_[1]<211 and 178<point_[2]<188  and len(surface_points_region3) < region_points:
            surface_points_region3.append(point_)
        elif len(surface_points_region1) >= region_points and len(surface_points_region2) >= region_points and len(surface_points_region3) >= region_points:
            is_ok = True

    surface_sampled = surface_points_region1 + surface_points_region2 + surface_points_region3
    surface_sampled = np.array(surface_sampled)
    np.savetxt(f'../../Tests/test1/surface_sampled_{i:02}.txt', surface_sampled, delimiter='\t', fmt='%.6f')   


# %% Visualization
vis = VisualizePoints()
vis.show_points_on_points(cropped_points_inner, surface_sampled, title='Sampled Surface Points')
# vis.set_data(fixed=cropped_points_inner)
# vis.show_points(vis.fixed, title='Cropped by outter bbox', size=1)
