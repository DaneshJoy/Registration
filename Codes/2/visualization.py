import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class VisualizePoints:
    
    def __init__(self):
        self.fig_single = plt.figure()
        self.ax_single = self.fig_single.gca(projection='3d')
        
        self.fig_double = plt.figure()
        self.ax1 = self.fig_double.add_subplot(121, projection='3d')
        self.ax2 = self.fig_double.add_subplot(122, projection='3d')
        
    def set_data(self, fixed, moving, landmarks):
        self.fixed = fixed
        self.moving = moving
        self.landmarks = landmarks
        
    def show_points(self, data, title, ax, color='b', marker='o', size=10):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=color, marker=marker, s=size)
        ax.set_title(title)
        self.show_bbox(self.moving, ax)
        # plt.show()
    
    def show_side_by_side(self):
        self.show_points(self.fixed, title='Image Points', color='blue', marker='.', size=1, ax=self.ax1)
        self.show_points(self.moving, title='Tracker Points', color='blue', marker='.', size=80, ax=self.ax2)
        self.show_points(self.landmarks, title='Tracker Points', color='red', marker='^', size=80, ax=self.ax2)
        self.ax2.set_xlim(self.ax1.get_xlim())
        self.ax2.set_ylim(self.ax1.get_ylim())
        self.ax2.set_zlim(self.ax1.get_zlim())
        
        # draw borders around subplots
        self.rect1 = plt.Rectangle(
            # (lower-left corner), width, height
            (0.02, 0.02), 0.48, 0.97, fill=False, color="k", lw=1, 
            zorder=1000, transform=self.fig_double.transFigure, figure=self.fig_double
        )
        self.rect2 = plt.Rectangle(
            # (lower-left corner), width, height
            (0.51, 0.02), 0.47, 0.97, fill=False, color="k", lw=1, 
            zorder=1000, transform=self.fig_double.transFigure, figure=self.fig_double
        )
        self.fig_double.patches.extend([self.rect1, self.rect2])
        self.fig_double.tight_layout()
        
        def on_move(event):
            if event.inaxes == self.ax1:
                self.ax2.view_init(elev=self.ax1.elev, azim=self.ax1.azim)
            elif event.inaxes == self.ax2:
                self.ax1.view_init(elev=self.ax2.elev, azim=self.ax2.azim)
            else:
                return

            self.fig_double.canvas.draw_idle()
        self.fig_double.canvas.mpl_connect('motion_notify_event', on_move)

        plt.show()

    def show_bbox(self, data, ax, padding=0, color='r', alpha=0.8):
        min_pts_outer = [np.min(data[:, [0]])-5,
                         np.min(data[:, [1]])-5,
                         np.min(data[:, [2]])-5]
        max_pts_outer = [np.max(data[:, [0]])+5,
                         np.max(data[:, [1]]),
                         np.max(data[:, [2]])+5]

        min_pts_inner = [np.min(data[:, [0]])+5,
                         np.min(data[:, [1]])+5,
                         np.min(data[:, [2]])+5]
        max_pts_inner = [np.max(data[:, [0]])-10,
                         np.max(data[:, [1]]),
                         np.max(data[:, [2]])-5]

        size_outer = [(max_pts_outer[0]-min_pts_outer[0]),
                      (max_pts_outer[1]-min_pts_outer[1]),
                      (max_pts_outer[2]-min_pts_outer[2])]
        
        center_outer = [min_pts_outer[0] + size_outer[0]//2,
                        min_pts_outer[1] + size_outer[1]//2,
                        min_pts_outer[2] + size_outer[2]//2]
        
        size_inner = [(max_pts_inner[0]-min_pts_inner[0]),
                      (max_pts_inner[1]-min_pts_inner[1]),
                      (max_pts_inner[2]-min_pts_inner[2])]
        
        center_inner = [min_pts_inner[0] + size_inner[0]//2,
                        min_pts_inner[1] + size_inner[1]//2,
                        min_pts_inner[2] + size_inner[2]//2]
        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o1 = [a - b / 2 for a, b in zip(center_outer, size_outer)]
        o2 = [a - b / 2 for a, b in zip(center_inner, size_inner)]
        # get the length, width, and height
        l1, w1, h1 = size_outer
        l2, w2, h2 = size_inner
        
        x1, y1, z1 = self.calculate_xyz(o1, l1, w1, h1)
        x2, y2, z2 = self.calculate_xyz(o2, l2, w2, h2)
        
        ax.plot_wireframe(np.array(x1), np.array(y1), np.array(z1),
                          color=color, alpha=alpha)
        ax.plot_wireframe(np.array(x2), np.array(y2), np.array(z2),
                          color='green', alpha=alpha)
        # return np.array(x), np.array(y), np.array(z)
    
    def calculate_xyz(self, o, l, w, h):
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
        return x, y, z
