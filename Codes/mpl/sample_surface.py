import numpy as np
from visualization import VisualizePoints
import random as rnd
from tqdm import tqdm

ex = 'ex4'
region_points = 20
cropped_points_outer = np.loadtxt(f'../../Tests/test1/{ex}/fixed_cropped_outer.txt')
cropped_points_inner = np.loadtxt(f'../../Tests/test1/{ex}/fixed_cropped_inner.txt')

# %% Random Sample Surface Points
for i in tqdm(range(100)):
    samples = []
    while True:
        point_ = rnd.choice(cropped_points_inner)
        for p in samples:
            if np.linalg.norm(p-point_) < 5:
                continue
        samples.append(point_)
        if len(samples) == region_points:
            break

    # samples = rnd.sample(list(cropped_points_inner), region_points)

    surface_sampled = np.array(samples)
    np.savetxt(f'../../Tests/test1/{ex}/surface_sampled_{i:02}.txt', surface_sampled, delimiter='\t', fmt='%.6f')   


# %% Visualization
vis = VisualizePoints()
vis.show_points_on_points(cropped_points_inner, surface_sampled, title='Sampled Surface Points')
# vis.set_data(fixed=cropped_points_inner)
# vis.show_points(vis.fixed, title='Cropped by outter bbox', size=1)
