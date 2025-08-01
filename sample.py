from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

def train_model(epochs=25, img_size=416, batch_size=8, patience=10):
    """
    Train a YOLOv8 model on the safety equipment dataset
    
    Args:
        epochs: Number of training epochs (reduced for i5 1335U CPU)
        img_size: Input image size (reduced for faster processing)
        batch_size: Batch size for training (reduced for CPU memory)
        patience: Early stopping patience
    
    Returns:
        Trained model and training results
    """
    print("Starting model training...")
    
    try:
        # Get the current directory where the script is located
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = current_dir / "HackByte_Dataset"
        
        # Check if dataset exists
        if not dataset_dir.exists():
            print(f"Error: Dataset directory {dataset_dir} not found")
            return None, None
            
        # Check if yolo_params.yaml exists
        yaml_path = current_dir / "yolo_params.yaml"
        if not yaml_path.exists():
            print(f"Error: YAML file {yaml_path} not found")
            return None, None
        
        # Load base YOLOv8 model
        model = YOLO("yolov8m.pt")  # Using small model for better performance on i5 CPU
        
        # Define training arguments with optimized hyperparameters
        args = {
            "data": str(yaml_path),
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch_size,
            "patience": patience,
            "device": 'cpu',  # Force CPU usage for i5 1335U
        # Hyperparameters for better performance in challenging conditions (optimized for CPU)
        "mosaic": 0.5,  # Reduced mosaic augmentation
        "mixup": 0.1,  # Reduced mixup augmentation
        "copy_paste": 0.1,  # Reduced copy-paste augmentation
        "degrees": 5.0,  # Reduced image rotation (+/- deg)
        "translate": 0.1,  # Image translation (+/- fraction)
        "scale": 0.1,  # Reduced image scale (+/- gain)
        "shear": 2.0,  # Reduced image shear (+/- deg)
        "perspective": 0.0001,  # Image perspective (+/- fraction)
        "flipud": 0.1,  # Image flip up-down (probability)
        "fliplr": 0.5,  # Image flip left-right (probability)
        "hsv_h": 0.015,  # Image HSV-Hue augmentation (fraction)
        "hsv_s": 0.2,  # Image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.2,  # Image HSV-Value augmentation (fraction)
        "optimizer": "AdamW",  # Optimizer (SGD, Adam, AdamW, etc.)
        "lr0": 0.001,  # Initial learning rate
        "lrf": 0.01,  # Final learning rate (fraction of lr0)
        "momentum": 0.937,  # SGD momentum/Adam beta1
        "weight_decay": 0.0005,  # Optimizer weight decay
        "warmup_epochs": 3.0,  # Warmup epochs (fractions ok)
        "warmup_momentum": 0.8,  # Warmup initial momentum
        "warmup_bias_lr": 0.1,  # Warmup initial bias lr
        "single_cls": False,  # Train multi-class data as single-class
    }
    
        # Train the model
        results = model.train(**args)
        
        print(f"Model training completed. Results saved to {results}")
        
        return model, results
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
def evaluate_model():
    """
    Evaluate the trained model on the test set and show metrics
    
    Returns:
        Model evaluation metrics
    """
    print("Evaluating model...")
    
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = current_dir / "HackByte_Dataset"
    yaml_path = current_dir / "yolo_params.yaml"
    
    if not yaml_path.exists():
        print(f"Error: YAML file {yaml_path} not found")
        return None
        
    data_yaml = str(yaml_path)
    
    # Find the best trained model
    detect_path = current_dir / "runs" / "detect"
    
    if not detect_path.exists():
        print("No 'runs/detect' directory found. Please train a model first.")
        return None
        
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    
    if not train_folders:
        print("No trained models found. Please train a model first.")
        return None
    
    # Get the latest training run
    latest_run = sorted(train_folders)[-1]
    model_path = detect_path / latest_run / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"Model weights not found at {model_path}. Please train a model first.")
        return None
    
    # Load the model
    model = YOLO(model_path)
    
    # Run validation on the test set
    metrics = model.val(data=data_yaml, split="test")
    
    print(f"Model evaluation completed. mAP50: {metrics.box.map50:.4f}")
    
    return metrics

def visualize_results(n_samples=5):
    """
    Visualize some prediction results from the test set
    
    Args:
        n_samples: Number of test images to visualize
    """
    print("Visualizing prediction results...")
    
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = current_dir / "HackByte_Dataset"
    yaml_path = current_dir / "yolo_params.yaml"
    
    if not yaml_path.exists():
        print(f"Error: YAML file {yaml_path} not found")
        return
    
    # Find the best trained model
    detect_path = current_dir / "runs" / "detect"
    
    if not detect_path.exists():
        print("No 'runs/detect' directory found. Please train a model first.")
        return
        
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    
    if not train_folders:
        print("No trained models found. Please train a model first.")
        return
    
    # Get the latest training run
    latest_run = sorted(train_folders)[-1]
    model_path = detect_path / latest_run / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"Model weights not found at {model_path}. Please train a model first.")
        return
        
    # Load the model
    model = YOLO(model_path)
    
    # Get test images
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            if 'test' in data and data['test']:
                test_dir = Path(dataset_dir) / data['test'] / 'images'
                if not test_dir.exists():
                    print(f"Test directory {test_dir} does not exist")
                    return
            else:
                print("No 'test' field found in yolo_params.yaml")
                return
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return
    
    # Get random sample of test images
    test_images = list(test_dir.glob('*.png'))
    if not test_images:
        print(f"No PNG images found in test directory {test_dir}")
        return
        
    np.random.shuffle(test_images)
    sample_images = test_images[:n_samples]
    
    # Make predictions and visualize
    plt.figure(figsize=(15, 15))
    for i, img_path in enumerate(sample_images):
        # Run prediction
        results = model.predict(str(img_path), conf=0.5)
        result = results[0]
        
        # Plot the prediction
        img = result.plot()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(3, 2, i+1)
        plt.imshow(img)
        plt.title(f"Test Image {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = current_dir / "results"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "prediction_samples.png")
    plt.close()
    
    print(f"Visualization saved to {output_dir / 'prediction_samples.png'}")

def main():
    """
    Main function to run the entire pipeline
    """
    print("Starting YOLOv8 object detection for safety equipment...")
    
    try:
        # Step 1: Train the model (reduced epochs for i5 1335U CPU)
        model, results = train_model(epochs=20)
        
        # Step 2: Evaluate the model
        metrics = evaluate_model()
        
        # Step 3: Visualize some results
        visualize_results(n_samples=6)
        
        # Print final metrics
        if metrics:
            print("\nFinal evaluation metrics:")
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.mp:.4f}")
            print(f"Recall: {metrics.box.mr:.4f}")
            
            # Save metrics to file
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            output_dir = current_dir / "results"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "metrics.txt", "w") as f:
                f.write(f"mAP50: {metrics.box.map50:.4f}\n")
                f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
                f.write(f"Precision: {metrics.box.mp:.4f}\n")
                f.write(f"Recall: {metrics.box.mr:.4f}\n")
        
        print("YOLOv8 object detection pipeline completed!")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
