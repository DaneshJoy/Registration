import random as rnd

import numpy as np
from scipy import linalg
from tqdm import tqdm

import vtk
from point_utils import draw_points, draw_two_pointclouds, xyz_to_ply
from vtk.util.numpy_support import vtk_to_numpy
# import reglib
from probreg import filterreg, l2dist_regs, cpd
from probreg import callbacks
import open3d as o3


def estimate_normals(pcd, params):
    pcd.estimate_normals(search_param=params)
    pcd.orient_normals_to_align_with_direction()


# %% Parameters
landmarkTransformType = "RigidBody"  # options RigidBody, Similarity, Affine
meanDistanceType = "RMS"  # options: RMS, Absolute Value
display_points = False
print_info = False
display_errors = False
print_matrix = False

ex = 'ex4'
algorithm = 'cpd'  # options: icp, filterreg, cpd, svr
start_by_matching_centroids = False

mean_errors = []
std_errors = []
min_errors = []
max_errors = []

mean_errors_landmark = []
std_errors_landmark = []
min_errors_landmark = []
max_errors_landmark = []

fixed_pts = np.loadtxt(f'../../Tests/test1/{ex}/fixed_cropped_inner.txt')
fixed = xyz_to_ply(fixed_pts)

landmark_pts = np.loadtxt(
    '../../Tests/test1/data/SurfaceRegOutput2/landmarks.txt')
landmarks = xyz_to_ply(landmark_pts)

for i in tqdm(range(100)):
    moving_pts = np.loadtxt(
        f'../../Tests/test1/{ex}/surface_sampled_{i:02}.txt')

    if display_points:
        # draw_points(fixed_pts)
        draw_two_pointclouds(fixed_pts, moving_pts, color1=[0, 1, 0])

    moving = xyz_to_ply(moving_pts)

    # random transform
    tr = vtk.vtkTransform()
    tr.Translate(rnd.uniform(-5, 5), rnd.uniform(-5, 5), rnd.uniform(-5, 5))
    tr.RotateX(rnd.uniform(-2, 2))

    tf1 = vtk.vtkTransformPolyDataFilter()
    tf1.SetTransform(tr)
    tf1.SetInputData(landmarks)
    tf1.Update()
    landmarks_displaced = tf1.GetOutput()
    landmark_pts_displaced = vtk_to_numpy(
        landmarks_displaced.GetPoints().GetData())

    tf2 = vtk.vtkTransformPolyDataFilter()
    tf2.SetTransform(tr)
    tf2.SetInputData(moving)
    tf2.Update()
    moving_displaced = tf2.GetOutput()
    moving_pts_displaced = vtk_to_numpy(moving_displaced.GetPoints().GetData())

    if display_points:
        draw_two_pointclouds(fixed_pts, moving_pts_displaced)

    if algorithm in ['filterreg', 'cpd', 'svr', 'gmmtree']:
        source = o3.geometry.PointCloud()
        target = o3.geometry.PointCloud()
        source.points = o3.utility.Vector3dVector(moving_pts_displaced)
        target.points = o3.utility.Vector3dVector(fixed_pts)
        # cbs = [callbacks.Open3dVisualizerCallback(source, target, keep_window=False)]
        cbs = []
        if algorithm == 'filterreg':
            tf_param, _, _ = filterreg.registration_filterreg(source, target,
                                                            objective_type='pt2pt',
                                                            sigma2=None,
                                                            update_sigma2=True,
                                                            callbacks=cbs)
        elif algorithm == 'cpd':
            tf_param, _, _ = cpd.registration_cpd(moving_pts_displaced, fixed_pts, callbacks=cbs)
        elif algorithm == 'svr':
            tf_param = l2dist_regs.registration_svr(source, target, callbacks=cbs)
        elif algorithm == 'gmmtree':
            tf_param, _ = gmmtree.registration_gmmtree(source, target, callbacks=cbs)

        trans1 = np.zeros([4, 4])
        trans1[-1, -1] = 1
        trans1[0:3, :-1] = tf_param.rot
        trans1[0:3, -1] = tf_param.t

    # fixed_pts = np.array(fixed_pts, dtype="float64")
    # moving_pts_displaced = np.array(moving_pts_displaced, dtype="float64")
    # trans = reglib.icp(source=moving_pts_displaced, target=fixed_pts, nr_iterations=30, epsilon=0.01,
    #                    inlier_threshold=0.05, distance_threshold=12.0, downsample=0, visualize=False)
    else:
        if print_info:
            print("Running ICP ----------------")
        # ============ run ICP ==============
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(moving_displaced)
        icp.SetTarget(fixed)

        if landmarkTransformType == "RigidBody":
            icp.GetLandmarkTransform().SetModeToRigidBody()
        elif landmarkTransformType == "Similarity":
            icp.GetLandmarkTransform().SetModeToSimilarity()
        elif landmarkTransformType == "Affine":
            icp.GetLandmarkTransform().SetModeToAffine()

        if meanDistanceType == "RMS":
            icp.SetMeanDistanceModeToRMS()
        elif meanDistanceType == "Absolute Value":
            icp.SetMeanDistanceModeToAbsoluteValue()

        # icp.DebugOn()
        icp.SetMaximumNumberOfIterations(30)
        if start_by_matching_centroids:
            icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        outputMatrix = vtk.vtkMatrix4x4()
        icp.GetMatrix(outputMatrix)
        # outputTrans.SetAndObserveMatrixTransformToParent(outputMatrix)
        matrix_np = np.eye(4)
        outputMatrix.DeepCopy(matrix_np.ravel(), outputMatrix)

    if algorithm in ['filterreg', 'cpd', 'svr', 'gmmtree']:
        outputMatrix2 = vtk.vtkMatrix4x4()
        for i in range(0, 4):
            for j in range(0, 4):
                outputMatrix2.SetElement(i, j, trans1[i, j])  
        t = vtk.vtkTransform()
        t.SetMatrix(outputMatrix2)
        # t.SetMatrix(icp.GetMatrix())
    else:
        t = icp

    if print_matrix:
        print(f'PCL Matrix:\n{trans1}')
        print(f'VTK Matrix:\n{matrix_np}')

    icpTransformFilter1 = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter1.SetTransform(t)
    # icpTransformFilter1.SetTransform(icp)
    icpTransformFilter1.SetInputData(landmarks_displaced)
    icpTransformFilter1.Update()
    transformedlandmarks = icpTransformFilter1.GetOutput()

    icpTransformFilter2 = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter2.SetTransform(t)
    # icpTransformFilter2.SetTransform(icp)
    icpTransformFilter2.SetInputData(moving_displaced)
    icpTransformFilter2.Update()
    transformedmoving = icpTransformFilter2.GetOutput()

    landmark_pts_registered = vtk_to_numpy(
        transformedlandmarks.GetPoints().GetData())
    moving_pts_registered = vtk_to_numpy(
        transformedmoving.GetPoints().GetData())
    if display_points:
        draw_two_pointclouds(fixed_pts, moving_pts_registered, moving_pts)
        draw_two_pointclouds(fixed_pts, landmark_pts_registered, landmark_pts)

    # %% Calculate errors
    errors_before = [
        linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(moving_pts, moving_pts_displaced)
    ]
    mean_err_before = np.mean(errors_before)
    std_err_before = np.std(errors_before)
    min_err_before = np.min(errors_before)
    max_err_before = np.max(errors_before)

    errors_before_landmark = [
        linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(landmark_pts, landmark_pts_displaced)
    ]
    mean_err_before_landmark = np.mean(errors_before_landmark)
    std_err_before_landmark = np.std(errors_before_landmark)
    min_err_before_landmark = np.min(errors_before_landmark)
    max_err_before_landmark = np.max(errors_before_landmark)

    errors_after = [
        linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(moving_pts, moving_pts_registered)
    ]
    mean_err_after = np.mean(errors_after)
    std_err_after = np.std(errors_after)
    min_err_after = np.min(errors_after)
    max_err_after = np.max(errors_after)

    errors_after_landmark = [
        linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(landmark_pts, landmark_pts_registered)
    ]
    mean_err_after_landmark = np.mean(errors_after_landmark)
    std_err_after_landmark = np.std(errors_after_landmark)
    min_err_after_landmark = np.min(errors_after_landmark)
    max_err_after_landmark = np.max(errors_after_landmark)

    if print_info:
        print('----- Errors before registration:')
        # print(f'\tAll: {errors}')
        print(f'\tMean/STD: {mean_err_before} ±{std_err_before}')
        print(f'\tMin: {min_err_before}')
        print(f'\tMax: {max_err_before}')
        print('---- Landmarks (TRE)')
        print(
            f'\tMean/STD: {mean_err_before_landmark} ±{std_err_before_landmark}')
        print(f'\tMin: {min_err_before_landmark}')
        print(f'\tMax: {max_err_before_landmark}')

    mean_errors.append(mean_err_after)
    std_errors.append(std_err_after)
    min_errors.append(min_err_after)
    max_errors.append(max_err_after)

    mean_errors_landmark.append(mean_err_after_landmark)
    std_errors_landmark.append(std_err_after_landmark)
    min_errors_landmark.append(min_err_after_landmark)
    max_errors_landmark.append(max_err_after_landmark)

    if print_info:
        print('----- Errors after registration:')
        # print(f'\tAll: {errors}')
        print('--- Surface')
        print(f'\tMean: {mean_err_after} ±{std_err_after}')
        print(f'\tMin: {min_err_after}')
        print(f'\tMax: {max_err_after}')
        print('--- Landmarks')
        print(f'\tMean: {mean_err_after_landmark} ±{std_err_after_landmark}')
        print(f'\tMin: {min_err_after_landmark}')
        print(f'\tMax: {max_err_after_landmark}')

    if display_errors:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        figure_size = (8, 6)
        min_err = 8
        max_err = 15
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection="3d")
        if not min_err:
            min_err = min_errors
        if not max_err:
            max_err = max_errors

        collection = ax.scatter(
            list(np.array(moving_pts_registered).T)[0],
            list(np.array(moving_pts_registered).T)[1],
            list(np.array(moving_pts_registered).T)[2],
            marker="o",
            c=errors_after,
            vmin=min_err,
            vmax=max_err,
            cmap=matplotlib.cm.jet,
            label="fixed points",
        )
        plt.colorbar(collection, shrink=0.8)
        plt.title('Registration Errors (in voxels)\n' +
                  f'mean: {mean_err_after:0.2f} ± {std_err_after:0.2f}\n' +
                  f'min: {min_err:0.2f}\n'
                  f'max: {max_err:0.2f}', x=0.7, y=1.05)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

print()
print('───● Average Results:')
# print(f'\tAll: {errors}')
print(' ├──● Surface')
print(f' │ ├──• Mean: {np.mean(mean_errors):4.2f} ±{np.mean(std_errors):.2f}')
print(f' │ ├──• Min: {np.mean(min_errors):5.2f}')
print(f' │ └──• Max: {np.mean(max_errors):5.2f}')
print(' └──● Landmark (TRE)')
print(
    f'   ├──• Mean: {np.mean(mean_errors_landmark):4.2f} ±{np.mean(std_errors_landmark):.2f}')
print(f'   ├──• Min: {np.mean(min_errors_landmark):5.2f}')
print(f'   └──• Max: {np.mean(max_errors_landmark):5.2f}')
print()
