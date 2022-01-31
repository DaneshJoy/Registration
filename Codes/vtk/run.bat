@ECHO OFF
CALL conda activate bv
python vtk_xyz_test.py "C:\Parsiss\MIS-4.6.7-a8-2201\SurfaceRegOutput0\OuterLayerPoints.txt.txt"
PAUSE