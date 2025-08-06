from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

def plot_confusion_matrix(model_path, data_yaml):
    """
    Plot and save confusion matrix for the model
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to the data yaml file
    """
    model = YOLO(model_path)
    
    # Run validation to get confusion matrix
    metrics = model.val(data=data_yaml, split="test", plots=True)
    
    # The confusion matrix is automatically saved in the runs/detect/val folder
    print(f"Confusion matrix saved in {os.path.dirname(model_path)}/../../val/")

def compare_predictions(model_path, image_paths, output_dir):
    """
    Compare predictions on multiple test images
    
    Args:
        model_path: Path to the trained model
        image_paths: List of image paths to make predictions on
        output_dir: Directory to save comparison images
    """
    model = YOLO(model_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions on all images
    for img_path in image_paths:
        # Original image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        results = model.predict(img_path, conf=0.25)
        result = results[0]
        
        # Get predicted image
        pred_img = result.plot()
        pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        
        # Create side-by-side comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img_rgb)
        plt.title("Prediction")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        img_name = os.path.basename(img_path)
        plt.savefig(os.path.join(output_dir, f"comparison_{img_name}.png"))
        plt.close()
    
    print(f"Comparison images saved to {output_dir}")

def plot_training_metrics(results_path):
    """
    Plot training metrics from the results.csv file
    
    Args:
        results_path: Path to the results.csv file
    """
    # Check if results file exists
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    # Read results
    import pandas as pd
    results = pd.read_csv(results_path)
    
    # Plot training metrics
    plt.figure(figsize=(15, 10))
    
    # Plot box loss
    plt.subplot(2, 2, 1)
    plt.plot(results['epoch'], results['train/box_loss'], label='train')
    plt.plot(results['epoch'], results['val/box_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.title('Box Loss')
    plt.legend()
    
    # Plot classification loss
    plt.subplot(2, 2, 2)
    plt.plot(results['epoch'], results['train/cls_loss'], label='train')
    plt.plot(results['epoch'], results['val/cls_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Loss')
    plt.title('Classification Loss')
    plt.legend()
    
    # Plot dfl loss (distribution focal loss)
    plt.subplot(2, 2, 3)
    plt.plot(results['epoch'], results['train/dfl_loss'], label='train')
    plt.plot(results['epoch'], results['val/dfl_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Distribution Focal Loss')
    plt.legend()
    
    # Plot mAP@0.5
    plt.subplot(2, 2, 4)
    plt.plot(results['epoch'], results['metrics/mAP50(B)'])
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.title('mAP@0.5')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(results_path)
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()
    
    print(f"Training metrics plot saved to {os.path.join(output_dir, 'training_metrics.png')}")

def analyze_class_performance(model_path, data_yaml):
    """
    Analyze the performance of each class
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to the data yaml file
    """
    model = YOLO(model_path)
    
    # Run validation to get per-class metrics
    metrics = model.val(data=data_yaml, split="test")
    
    # Get class names
    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)
        class_names = data['names']
    
    # Extract per-class metrics
    per_class_ap = metrics.box.ap_class
    per_class_precision = metrics.box.p_class
    per_class_recall = metrics.box.r_class
    
    # Create bar plot for class performance
    plt.figure(figsize=(12, 8))
    
    # Class AP@0.5
    x = np.arange(len(class_names))
    width = 0.2
    
    plt.bar(x - width, per_class_ap, width, label='AP@0.5:0.95')
    plt.bar(x, per_class_precision, width, label='Precision')
    plt.bar(x + width, per_class_recall, width, label='Recall')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Performance')
    plt.xticks(x, class_names)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(os.path.dirname(model_path))
    plt.savefig(os.path.join(output_dir, "class_performance.png"))
    plt.close()
    
    print(f"Class performance plot saved to {os.path.join(output_dir, 'class_performance.png')}")
    
    # Print per-class metrics
    print("\nPer-Class Performance:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  AP@0.5:0.95: {per_class_ap[i]:.4f}")
        print(f"  Precision: {per_class_precision[i]:.4f}")
        print(f"  Recall: {per_class_recall[i]:.4f}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Model Analysis')
    parser.add_argument('--analyze', action='store_true', help='Analyze model performance')
    parser.add_argument('--compare', action='store_true', help='Compare predictions')
    parser.add_argument('--metrics', action='store_true', help='Plot training metrics')
    parser.add_argument('--confusion', action='store_true', help='Plot confusion matrix')
    parser.add_argument('--model', type=str, help='Path to model weights file (best.pt)')
    args = parser.parse_args()
    
    # Get current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = current_dir / "HackByte_Dataset"
    yaml_path = current_dir / "yolo_params.yaml"
    
    if not yaml_path.exists():
        print(f"Error: YAML file {yaml_path} not found")
        print("Please make sure the yolo_params.yaml file exists in the main directory")
        return
        
    data_yaml = str(yaml_path)
    
    # Check if the model was provided directly
    if args.model and os.path.exists(args.model):
        print(f"Using provided model: {args.model}")
        model_path = args.model
        # Get the parent directory for results
        results_dir = os.path.dirname(os.path.dirname(model_path))
        results_path = os.path.join(results_dir, "results.csv")
    else:
        # Check for runs directory - look in the current directory first, not in dataset_dir
        runs_path = current_dir / "runs"
        if not os.path.exists(runs_path):
            # If not found, try in dataset_dir as fallback
            runs_path = dataset_dir / "runs"
            if not os.path.exists(runs_path):
                print(f"Error: No 'runs' directory found at {current_dir}/runs or {dataset_dir}/runs")
                print("Please train a model first using sample.py or provide a model path with --model")
                print("Example: python analyze_model.py --model path/to/best.pt")
                return
        
        # Check for detect directory
        detect_path = runs_path / "detect"
        if not os.path.exists(detect_path):
            print(f"Error: No 'detect' directory found at {detect_path}")
            print("Please train a model first using sample.py or provide a model path with --model")
            return
        
        # Find training folders
        train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(os.path.join(detect_path, f)) and f.startswith("train")]
        
        if not train_folders:
            print(f"Error: No training folders found in {detect_path}")
            print("Please train a model first using sample.py")
            print("Run: python sample.py")
            return
        
        # Get the latest training run
        latest_run = sorted(train_folders)[-1]
        weights_dir = os.path.join(detect_path, latest_run, "weights")
        
        if not os.path.exists(weights_dir):
            print(f"Error: No weights directory found at {weights_dir}")
            print("The model training may not have completed successfully.")
            return
        
        model_path = os.path.join(weights_dir, "best.pt")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("The model training may not have completed successfully.")
            return
            
        results_path = os.path.join(detect_path, latest_run, "results.csv")
        
        print(f"Using latest trained model: {model_path}")
    
    # Convert model_path to Path object
    model_path = Path(model_path)
    results_path = Path(results_path) if os.path.exists(results_path) else None
    
    # Get test images
    try:
        with open(data_yaml, 'r') as file:
            data = yaml.safe_load(file)
            if 'test' in data and data['test']:
                test_dir_path = data['test']
                # Try to find test images in two possible locations
                test_dir1 = Path(dataset_dir) / test_dir_path / 'images'
                test_dir2 = Path(current_dir) / test_dir_path / 'images'
                
                # Check which path exists
                if os.path.exists(test_dir1):
                    test_dir = test_dir1
                elif os.path.exists(test_dir2):
                    test_dir = test_dir2
                else:
                    # Try directly using the path from YAML (might be absolute)
                    test_dir = Path(test_dir_path) / 'images'
                
                if os.path.exists(test_dir):
                    # Get sample test images
                    test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
                    if test_images:
                        np.random.shuffle(test_images)
                        sample_images = [str(img) for img in test_images[:5]]
                    else:
                        print(f"Warning: No images found in {test_dir}")
                        sample_images = []
                else:
                    print(f"Warning: Test directory {test_dir} does not exist")
                    sample_images = []
            else:
                print("Warning: 'test' field not found in YAML config")
                sample_images = []
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        sample_images = []
    
    # Output directory for visualizations
    output_dir = current_dir / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # Create a flag for whether any analysis was performed
    analysis_performed = False
    
    # Run requested analyses
    if args.analyze or not any([args.analyze, args.compare, args.metrics, args.confusion]):
        print("\nAnalyzing class performance...")
        try:
            analyze_class_performance(model_path, data_yaml)
            analysis_performed = True
        except Exception as e:
            print(f"Error analyzing class performance: {e}")
    
    if sample_images and (args.compare or not any([args.analyze, args.compare, args.metrics, args.confusion])):
        print("\nGenerating prediction comparisons...")
        try:
            compare_predictions(model_path, sample_images, str(output_dir / "comparisons"))
            analysis_performed = True
        except Exception as e:
            print(f"Error comparing predictions: {e}")
    elif args.compare:
        print("No sample images available for comparison")
    
    if results_path and os.path.exists(results_path) and (args.metrics or not any([args.analyze, args.compare, args.metrics, args.confusion])):
        print("\nPlotting training metrics...")
        try:
            plot_training_metrics(results_path)
            analysis_performed = True
        except Exception as e:
            print(f"Error plotting training metrics: {e}")
    elif args.metrics:
        print(f"Training metrics file not found: {results_path}")
    
    if args.confusion or not any([args.analyze, args.compare, args.metrics, args.confusion]):
        print("\nGenerating confusion matrix...")
        try:
            plot_confusion_matrix(model_path, data_yaml)
            analysis_performed = True
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            
    if analysis_performed:
        print("\nAnalysis completed!")
    else:
        print("\nNo analysis was performed due to missing files or errors.")
        print("Please train a model first using: python sample.py")
    
    print("Analysis completed!")

if __name__ == "__main__":
    main()
