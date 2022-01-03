import numpy as np
from point_utils import draw_points, draw_two_pointclouds, xyz_to_ply
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import random as rnd
from scipy import linalg
from tqdm import tqdm

# %% Parameters
landmarkTransformType = "RigidBody"
meanDistanceType = "RMS"
display_points = True
display_errors = False
print_info = False

mean_errors = []
std_errors = []
min_errors = []
max_errors = []

fixed_pts = np.loadtxt('../../Tests/test1/fixed_cropped_inner.txt')

for i in tqdm(range(100)):
    moving_pts = np.loadtxt(f'../../Tests/test1/surface_sampled_{i:02}.txt')

    if display_points:
        # draw_points(fixed_pts)
        draw_two_pointclouds(fixed_pts, moving_pts)

    moving = xyz_to_ply(moving_pts)
    fixed = xyz_to_ply(fixed_pts)

    # random transform
    tr = vtk.vtkTransform()
    tr.Translate(rnd.uniform(-1,5), rnd.uniform(-1,5), rnd.uniform(-1,5))
    tr.RotateX(rnd.uniform(-2,2))

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(tr)
    tf.SetInputData(moving)
    tf.Update()
    moving = tf.GetOutput()
    moving_pts_displaced = vtk_to_numpy(moving.GetPoints().GetData())

    if display_points:
        draw_two_pointclouds(fixed_pts, moving_pts_displaced)

    if print_info:
        print("Running ICP ----------------")
    # ============ run ICP ==============
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(moving)
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
    icp.SetMaximumNumberOfIterations(40)
    # icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    outputMatrix = vtk.vtkMatrix4x4()
    icp.GetMatrix(outputMatrix)
    # outputTrans.SetAndObserveMatrixTransformToParent(outputMatrix)
    matrix_np = np.eye(4)
    outputMatrix.DeepCopy(matrix_np.ravel(), outputMatrix)

    if print_info:
        print('Finished icp')
        print(f'Matrix:\n{matrix_np}')
        
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(moving)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    transformedmoving = icpTransformFilter.GetOutput()

    moving_pts_registered = vtk_to_numpy(transformedmoving.GetPoints().GetData())
    if display_points:
        draw_two_pointclouds(fixed_pts, moving_pts_registered, moving_pts)

    # %% Calculate errors
    errors = [
        linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(moving_pts, moving_pts_displaced)
    ]
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)
    
    errors_before = [
        linalg.norm(np.array(p_fixed) - np.array(p_moving))
        for p_fixed, p_moving in zip(moving_pts, moving_pts_registered)
    ]
    mean_err_before = np.mean(errors_before)
    std_err_before = np.std(errors_before)
    min_err_before = np.min(errors_before)
    max_err_before = np.max(errors_before)
    
    if print_info:
        print('----- Errors before registration:')
        # print(f'\tAll: {errors}')
        print(f'\tMean/STD: {mean_err_before} ± {std_err_before}')
        print(f'\tMin: {min_err_before}')
        print(f'\tMax: {max_err_before}')
    
    mean_errors.append(mean_err)
    std_errors.append(std_err)
    min_errors.append(min_err)
    max_errors.append(max_err)

    if print_info:
        print('----- Errors after registration:')
        # print(f'\tAll: {errors}')
        print(f'\tMean/STD: {mean_err} ± {std_err}')
        print(f'\tMin: {min_err}')
        print(f'\tMax: {max_err}')

    if display_errors:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import matplotlib

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
            c=errors,
            vmin=min_err,
            vmax=max_err,
            cmap=matplotlib.cm.jet,
            label="fixed points",
        )
        plt.colorbar(collection, shrink=0.8)
        plt.title('Registration Errors (in voxels)\n' +
                f'mean: {mean_err:0.2f} ± {std_err:0.2f}\n' +
                f'min: {min_err:0.2f}\n'
                f'max: {max_err:0.2f}', x=0.7, y=1.05)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

print('└──• Average Results:')
# print(f'\tAll: {errors}')
print(f' ├──• Mean ± STD: {np.mean(mean_errors):>5.2f} ± {np.mean(std_errors):>5.2f}')
print(f' ├──• Min: {np.mean(min_errors):>12.2f}')
print(f' └──• Max: {np.mean(max_errors):>12.2f}')