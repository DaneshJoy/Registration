from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')

# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

    

# heads of the arrows with adjusted arrow head length
ax.quiver([-1, 0, 1], [-1, -1, 0], [0, 1, -1],
          [1, 0, 0], [0, 1, 0], [0, 0, 1],
          length=2, normalize=True,
          color='r', arrow_length_ratio=0.15)

ax.text(0, -1, 0, '0', zdir='x', fontsize=16)
ax.text(0, 0, 1, '1', zdir='x', fontsize=16)
ax.text(1, 0, 0, '2', zdir='x', fontsize=16)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

ax.set_axis_off()
ax.set_title('3D points Axis and Directions')

plt.show()