@ECHO OFF
ECHO.
CALL conda activate bv
python reg_test.py
ECHO.
PAUSE