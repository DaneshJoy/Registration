import numpy as np
from point_utils import draw_points, draw_two_pointclouds, xyz_to_ply
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from random import random as rnd


landmarkTransformType = "RigidBody"
meanDistanceType = "RMS"

target_pts = np.loadtxt('../../Tests/test1/fixed_cropped_inner.txt')
source_pts = np.loadtxt('../../Tests/test1/surface_sampled.txt')
    
# draw_points(target_pts)
draw_two_pointclouds(target_pts, source_pts)

source = xyz_to_ply(source_pts)
target = xyz_to_ply(target_pts)

# transform
tr = vtk.vtkTransform()
tr.Translate(rnd(), rnd(), rnd())
tr.RotateX(1+rnd())

tf = vtk.vtkTransformPolyDataFilter()
tf.SetTransform(tr)
tf.SetInputData(source)
tf.Update()
source = tf.GetOutput()
source_pts = vtk_to_numpy(source.GetPoints().GetData())
draw_two_pointclouds(target_pts, source_pts)

print("Running ICP ----------------")
# ============ run ICP ==============
icp = vtk.vtkIterativeClosestPointTransform()
icp.SetSource(source)
icp.SetTarget(target)

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
icp.SetMaximumNumberOfIterations(20)
icp.StartByMatchingCentroidsOn()
icp.Modified()
icp.Update()

outputMatrix = vtk.vtkMatrix4x4()
icp.GetMatrix(outputMatrix)
# outputTrans.SetAndObserveMatrixTransformToParent(outputMatrix)
matrix_np = np.eye(4)
outputMatrix.DeepCopy(matrix_np.ravel(), outputMatrix)

print('Finished icp')
print(f'Matrix:\n{matrix_np}')
icpTransformFilter = vtk.vtkTransformPolyDataFilter()
icpTransformFilter.SetInputData(source)
icpTransformFilter.SetTransform(icp)
icpTransformFilter.Update()

transformedSource = icpTransformFilter.GetOutput()

transformedSource_points = vtk_to_numpy(transformedSource.GetPoints().GetData())
draw_two_pointclouds(target_pts, transformedSource_points)
