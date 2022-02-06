import numpy as np
from point_utils import draw_points, draw_two_pointclouds, xyz_to_ply
import vtk
from vtk.util.numpy_support import vtk_to_numpy


landmarkTransformType = "RigidBody"
meanDistanceType = "RMS"
display_points = True
display_errors = False
print_info = True

moving_pts = np.loadtxt(r'C:\Parsiss\MIS-4.6.7-a8-2201\SurfaceRegOutput1\surfacePoints_Out3.txt')
fixed_pts = np.loadtxt(r'C:\Parsiss\misSoloutionBiopsyLog\CreateMISRelease_local\SurfaceRegOutput0\OuterLayerPoints.txt',
                       delimiter=',')

if display_points:
    draw_points(fixed_pts, point_size=1, color=[0, 1, 0])
    # draw_two_pointclouds(fixed_pts, moving_pts, point_size=1)

    
fixed = xyz_to_ply(fixed_pts)
moving = xyz_to_ply(moving_pts)

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
    draw_two_pointclouds(fixed_pts, moving_pts, moving_pts_registered)
