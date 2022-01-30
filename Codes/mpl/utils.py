import numpy as np


def bounding_box(points, min_pts=None,  max_pts=None, pad=None):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_pts, max_pts: float tuple
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_pts and infinite for the max_pts.

    pad: int tuple
        The padding (positive or negative) to adjust the edges of the bounding box
        Expected format: (x_start, y_start, z_start, x_end, y_end, z_end)
        the default values are 0 (no padding).

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    if min_pts is None:
        min_pts = (-np.inf, -np.inf, -np.inf)
    if max_pts is None:
        max_pts = (np.inf, np.inf,  np.inf)
    if pad is None:
        pad = (0, 0, 0, 0, 0, 0)

    bound_x = np.logical_and(points[:, 0] > min_pts[0]+pad[0], points[:, 0] < max_pts[0]+pad[3])
    bound_y = np.logical_and(points[:, 1] > min_pts[1]+pad[1], points[:, 1] < max_pts[1]+pad[4])
    bound_z = np.logical_and(points[:, 2] > min_pts[2]+pad[2], points[:, 2] < max_pts[2]+pad[5])

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter
