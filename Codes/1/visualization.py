import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class VisualizePoints:

    def __init__(self):
        self.fig = plt.figure()

    def vis_one(self, X, title, color, marker, size, ax):
        # ax.cla()
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker, s=size)
        ax.set_title(title)

    def show_data(self, image_coords, tracker_coords, lamdmarks_coords):
        self.target_coords = image_coords
        self.source_coords = tracker_coords
        self.landmark_coords = lamdmarks_coords
        self.axes1 = self.fig.add_subplot(121, projection='3d')
        self.vis_one(self.target_coords, title='Image Points', color='red', marker='+', size=1, ax=self.axes1)

        self.axes2 = self.fig.add_subplot(122, projection='3d')
        self.vis_one(self.source_coords, title='Tracker Points', color='blue', marker='.', size=80, ax=self.axes2)
        self.vis_one(self.landmark_coords, title='Tracker Points', color='blue', marker='*', size=80, ax=self.axes2)
        self.axes2.set_xlim(self.axes1.get_xlim())
        self.axes2.set_ylim(self.axes1.get_ylim())
        self.axes2.set_zlim(self.axes1.get_zlim())

        # draw borders around subplots
        self.rect1 = plt.Rectangle(
            # (lower-left corner), width, height
            (0.02, 0.02), 0.48, 0.97, fill=False, color="k", lw=1, 
            zorder=1000, transform=self.fig.transFigure, figure=self.fig
        )
        self.rect2 = plt.Rectangle(
            # (lower-left corner), width, height
            (0.51, 0.02), 0.47, 0.97, fill=False, color="k", lw=1, 
            zorder=1000, transform=self.fig.transFigure, figure=self.fig
        )
        self.fig.patches.extend([self.rect1, self.rect2])
        self.fig.tight_layout()

        def on_move(event):
            if event.inaxes == self.axes1:
                self.axes2.view_init(elev=self.axes1.elev, azim=self.axes1.azim)
            elif event.inaxes == self.axes2:
                self.axes1.view_init(elev=self.axes2.elev, azim=self.axes2.azim)
            else:
                return

            self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_connect('motion_notify_event', on_move)

        plt.show()

    def visualize_reg(self, iteration, error, X, Y, ax):
        # self.fig.delaxes(self.axes1)
        # self.fig.delaxes(self.axes2)
        # self.fig.patches = []
        ax.cla()
        ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', marker='+', label='Target')
        ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', marker='^', label='Source', s=80)
        ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
            iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.draw()
        plt.pause(0.001)


def cuboid_data(center, size):
    """
    Create a data array for cuboid plotting.


    ============= ================================================
    Argument      Description
    ============= ================================================
    center        center of the cuboid, triple
    size          size of the cuboid, triple, (x_length,y_width,z_height)
    :type size: tuple, numpy.array, list
    :param size: size of the cuboid, triple, (x_length,y_width,z_height)
    :type center: tuple, numpy.array, list
    :param center: center of the cuboid, triple, (x,y,z)
    """

    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return np.array(x), np.array(y), np.array(z)


def test():
    center = [0, 0, 0]
    length = 32 * 2
    width = 50 * 2
    height = 100 * 2
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = cuboid_data(center, (length, width, height))
    ax.plot_wireframe(X, Y, Z, color='r', rstride=1, cstride=1, alpha=0.9)
    ax.set_xlabel('X')
    ax.set_xlim(-100, 100)
    ax.set_ylabel('Y')
    ax.set_ylim(-100, 100)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)
    plt.show()


if __name__ == '__main__':
    test()