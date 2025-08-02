# YOLOv8 Safety Equipment Detection

This project implements a YOLOv8 model to detect three classes of safety equipment:
- Toolbox
- Oxygen Tank
- Fire Extinguisher

The model is trained to handle challenging conditions including:
1. Varying lighting conditions
2. Occlusions and object overlap
3. Diverse camera angles and distances

## Optimized for i5 1335U CPU

The model parameters have been adjusted to run efficiently on an i5 1335U CPU:
- Using YOLOv8s (medium) model instead large variants
- Reduced image size (416px vs. 640px)
- Smaller batch size (8 vs. 16)
- Fewer training epochs (20 vs. 50)
- Reduced data augmentation complexity
- Early stopping with patience=10

## Project Structure

```
.
├── data/   (from the falcon dataset moe to fata folder here)
│   ├── test/
│   ├── train/
│   └── val/
├── HackByte_Dataset/
│   ├── ENV_SETUP/
│   ├── classes.txt
│   ├── predict.py
│   ├── train.py
│   └── visualize.py
├── predictions/
│   ├── images/
│   └── labels/
├── runs/
│   └── detect/
├── test_case/
├── analyze_model.py
├── predict_custom.py
├── README.md
├── sample.py
├── yolo_params.yaml
├── yolo_safety_detector.bat
└── yolov8m.pt
```

## Required Libraries

- PyTorch
- Torchvision
- CUDA 11.8
- Ultralytics
- OpenCV

## Running the Model

1. Use the `yolo_safety_detector.bat` file to run the model with different settings:
   - Default setting (20 epochs)
   - Quick test (5 epochs)
   - Evaluation only (skip training)
   - Custom number of epochs

2. Alternatively, run the Python script directly:
   ```
   python sample.py
   ```

## Expected Training Time

On an i5 1335U CPU, expect:
- Full training (20 epochs): ~4-6 hours
- Quick test (5 epochs): ~1-1.5 hours
- Evaluation only: ~10-15 minutes

## Results

After training, the model will generate:
- mAP50 score (primary metric for object detection)
- Visualization of predictions on test images
- Detailed metrics report (precision, recall, mAP)

Results are saved in the `results/` directory.

## Analysis

For deeper analysis of the model's performance, run:
```
python analyze_model.py
```

This will generate:
- Per-class performance metrics
- Confusion matrix
- Training curves
- Side-by-side prediction comparisons
