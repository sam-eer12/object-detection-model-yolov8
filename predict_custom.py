from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import argparse
import glob

def predict_and_save(model, image_path, output_dir, conf_threshold=0.5):
    """
    Perform prediction on an image and save the results
    
    Args:
        model: The YOLO model
        image_path: Path to the image
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for predictions
    """
    # Create output directories if they don't exist
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image filename
    img_name = Path(image_path).name
    output_path_img = images_output_dir / img_name
    output_path_txt = labels_output_dir / Path(img_name).with_suffix('.txt').name
    
    # Perform prediction
    results = model.predict(image_path, conf=conf_threshold)
    result = results[0]
    
    # Draw boxes on the image
    img = result.plot()
    
    # Save the result image
    cv2.imwrite(str(output_path_img), img)
    
    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            # Extract the class id and bounding box coordinates
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywh[0].tolist()
            
            # Write bbox information in the format [class_id, x_center, y_center, width, height]
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
    
    # Print detection information
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]
        confidence = float(box.conf)
        detections.append(f"{class_name} ({confidence:.2f})")
    
    if detections:
        print(f"Detected: {', '.join(detections)}")
    else:
        print("No objects detected")
    
    return output_path_img, output_path_txt

def main():
    parser = argparse.ArgumentParser(description="Run predictions with trained YOLOv8 model")
    parser.add_argument("--model", type=str, help="Path to the model weights file (best.pt)")
    parser.add_argument("--input", type=str, help="Path to input image or directory")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)")
    parser.add_argument("--test_dir", action="store_true", help="Run on the test directory specified in yolo_params.yaml")
    
    args = parser.parse_args()
    
    # Get the current directory
    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(this_dir)
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Find the best model automatically
        detect_path = this_dir / "runs" / "detect"
        if not detect_path.exists():
            print(f"Error: No 'runs/detect' directory found at {detect_path}")
            print("Please train a model first or specify a model path with --model")
            return
        
        train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
        if len(train_folders) == 0:
            print("No training folders found. Please train a model first.")
            return
        
        # Get the latest training folder
        latest_run = sorted(train_folders)[-1]
        model_path = detect_path / latest_run / "weights" / "best.pt"
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print(f"Model loaded successfully. Classes: {model.names}")
    
    # Output directory
    output_dir = this_dir / "predictions"
    
    # Determine input images
    if args.test_dir:
        # Use the test directory from yolo_params.yaml
        with open(this_dir / 'yolo_params.yaml', 'r') as file:
            data = yaml.safe_load(file)
            if 'test' in data and data['test'] is not None:
                test_dir = Path(data['test']) / 'images'
                if not test_dir.exists() or not test_dir.is_dir():
                    print(f"Error: Test directory {test_dir} does not exist or is not a directory")
                    return
                
                # Process all images in the test directory
                print(f"Running predictions on test images in {test_dir}...")
                image_files = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg'))
                
                if not image_files:
                    print(f"No image files found in {test_dir}")
                    return
                
                for img_path in image_files:
                    print(f"Predicting on {img_path.name}...")
                    output_img, output_txt = predict_and_save(model, img_path, output_dir, args.conf)
                    print(f"Saved prediction to {output_img}")
                    print("\n")
                    print("\n")
                
                # Run validation
                print("\nRunning validation on test set...")
                metrics = model.val(data=this_dir / 'yolo_params.yaml', split="test")
                print(f"\nTest Results:")
                print(f"mAP50: {metrics.box.map50:.4f}")
                print(f"mAP50-95: {metrics.box.map:.4f}")
                print(f"Precision: {metrics.box.mp:.4f}")
                print(f"Recall: {metrics.box.mr:.4f}")
            else:
                print("No test field found in yolo_params.yaml")
                return
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            # Single image
            if input_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                print(f"Error: Unsupported image format. Supported formats: .png, .jpg, .jpeg")
                return
            
            print(f"Predicting on single image: {input_path}")
            output_img, output_txt = predict_and_save(model, input_path, output_dir, args.conf)
            print(f"Saved prediction to {output_img}")
            
        elif input_path.is_dir():
            # Directory of images
            print(f"Predicting on all images in directory: {input_path}")
            image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg'))
            
            if not image_files:
                print(f"No image files found in {input_path}")
                return
            
            for img_path in image_files:
                print(f"Predicting on {img_path.name}...")
                output_img, output_txt = predict_and_save(model, img_path, output_dir, args.conf)
                print(f"Saved prediction to {output_img}")
        else:
            print(f"Error: Input path {input_path} does not exist")
            return
    else:
        # No input specified, show usage
        print("Error: Please specify either --input for a custom image/directory or --test_dir to use the test set")
        print("Example usage:")
        print(f"  python predict_custom.py --test_dir")
        print(f"  python predict_custom.py --input path/to/image.jpg")
        print(f"  python predict_custom.py --input path/to/images/folder --conf 0.6")
        return
    
    print(f"\nAll predictions saved to {output_dir}")

if __name__ == "__main__":
    main()
