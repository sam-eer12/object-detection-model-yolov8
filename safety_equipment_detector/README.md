# YOLOv8 Safety Equipment Detector 
 
This model detects three types of safety equipment: 
- Toolbox 
- Oxygen Tank 
- Fire Extinguisher 
 
## Installation 
 
```bash 
pip install ultralytics opencv-python 
``` 
 
## Usage 
 
```bash 
python predict.py --image examples/sample_images/your_image.jpg 
``` 
 
For custom confidence threshold: 
 
```bash 
python predict.py --image examples/sample_images/your_image.jpg --conf 0.6 
``` 
