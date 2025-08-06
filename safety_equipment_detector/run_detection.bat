@echo off 
echo YOLOv8 Safety Equipment Detector 
echo ============================ 
echo. 
set /p image_path=Enter path to image: 
python predict.py --image "%image_path%" 
pause 
