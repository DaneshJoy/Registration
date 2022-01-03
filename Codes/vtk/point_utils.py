import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray


def draw_points(xyz_points):
    # Renderer
    # renderer = vtk.vtkRenderer()
    renderer = vtk.vtkOpenGLRenderer()
    # renderer.SetBackground(.2, .3, .4)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
             
    poly = xyz_to_ply(xyz_points)
    pmap = vtk.vtkPolyDataMapper()
    pmap.SetInputDataObject(poly)

    # actor
    points = vtk.vtkActor()
    points.SetMapper(pmap)
    points.GetProperty().SetPointSize(3)
    points.GetProperty().SetColor(0.2,0.5,1) # (R,G,B)

    # assign actor to the renderer
    renderer.AddActor(points)
    # renderer.Render()
    # renderWindowInteractor.GetRenderWindow().AddRenderer(renderer)
    renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    # interactor.ReInitialize()
    # Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("XYZ Data Viewer")
    renderWindowInteractor.Start()
    # self.ren.ResetCameraClippingRange()
    
def draw_two_pointclouds(fixed, moving1, moving2=[]):
    # Renderer
    # renderer = vtk.vtkRenderer()
    renderer = vtk.vtkOpenGLRenderer()
    # renderer.SetBackground(.2, .3, .4)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
             
    poly_fixed = xyz_to_ply(fixed)
    pmap_fixed = vtk.vtkPolyDataMapper()
    pmap_fixed.SetInputDataObject(poly_fixed)

    # actor
    points_fixed = vtk.vtkActor()
    points_fixed.SetMapper(pmap_fixed)
    points_fixed.GetProperty().SetPointSize(2)
    points_fixed.GetProperty().SetColor(0.3,0.4,0.4) # (R,G,B)
    
    poly_moving1 = xyz_to_ply(moving1)
    pmap_moving1 = vtk.vtkPolyDataMapper()
    pmap_moving1.SetInputDataObject(poly_moving1)

    # actor
    points_moving1 = vtk.vtkActor()
    points_moving1.SetMapper(pmap_moving1)
    points_moving1.GetProperty().SetPointSize(6)
    points_moving1.GetProperty().SetColor(0.9,0.1,0.1) # (R,G,B)

    if moving2 != []:
        poly_moving2 = xyz_to_ply(moving2)
        pmap_moving2 = vtk.vtkPolyDataMapper()
        pmap_moving2.SetInputDataObject(poly_moving2)

        # actor
        points_moving2 = vtk.vtkActor()
        points_moving2.SetMapper(pmap_moving2)
        points_moving2.GetProperty().SetPointSize(6)
        points_moving2.GetProperty().SetColor(0.1,0.9,0.1) # (R,G,B)
        
    # assign actor to the renderer
    renderer.AddActor(points_fixed)
    renderer.AddActor(points_moving1)
    if moving2 != []:
        renderer.AddActor(points_moving2)
    # renderer.Render()
    # renderWindowInteractor.GetRenderWindow().AddRenderer(renderer)
    renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    # interactor.ReInitialize()
    # Begin Interaction
    renderWindow.Render()
    renderWindow.SetWindowName("XYZ Data Viewer")
    renderWindowInteractor.Start()
    # self.ren.ResetCameraClippingRange()


def xyz_to_ply(xyz_points):
    pts = vtk.vtkPoints()
    conn = vtk.vtkCellArray()
    poly = vtk.vtkPolyData()
    nPoints = len(xyz_points)

    for i in range(0,nPoints):
        # pos = xyz_points[:,:,i]
        pos = xyz_points[i]
        pts.InsertNextPoint(pos[0], pos[1], pos[2])

    cells = np.hstack((np.ones((nPoints, 1)),
                       np.arange(nPoints).reshape(-1, 1)))
    cells = np.ascontiguousarray(cells, dtype=np.int64)

    conn.SetCells(nPoints, numpy_to_vtkIdTypeArray(cells, deep=True))

    poly.SetPoints(pts)
    poly.SetVerts(conn)

    
    return poly

    # flipTrans = vtk.vtkTransform()
    # flipTrans.Scale(-1,-1,1)
    # flipFilt = vtk.vtkTransformPolyDataFilter()
    # # flipFilt = vtk.vtkTransformFilter()
    # flipFilt.SetTransform(flipTrans)
    # flipFilt.SetInputData(poly)
    # flipFilt.Update()
    # pmap.SetInputDataObject(flipFilt.GetOutput())