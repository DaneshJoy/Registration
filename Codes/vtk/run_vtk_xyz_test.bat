@ECHO OFF
CALL conda activate bv
python vtk_xyz_test.py "D:\Parsiss\Registration\Tests\test1\data\SurfaceRegOutput2\OuterLayerPoints.txt"
PAUSE