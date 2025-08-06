from ultralytics import YOLO 
import cv2 
import argparse 
import os 
 
def predict_image(model_path, image_path, conf=0.5): 
    """Run prediction on an image and display/save results""" 
    # Load model 
    model = YOLO(model_path) 
    # Run prediction 
    results = model.predict(image_path, conf=conf) 
    result = results[0] 
    # Draw predictions on image 
    img = result.plot() 
    # Save prediction 
    output_path = os.path.join(os.path.dirname(image_path), "prediction_" + os.path.basename(image_path)) 
    cv2.imwrite(output_path, img) 
    # Display image 
    cv2.imshow("Prediction", img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    # Print detections 
    print(f"Prediction saved to {output_path}") 
    for box in result.boxes: 
        cls_id = int(box.cls) 
        class_name = model.names[cls_id] 
        confidence = float(box.conf) 
        x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]] 
        print(f"Detected {class_name} (confidence: {confidence:.2f}) at coordinates: [{x1}, {y1}, {x2}, {y2}]") 
 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run inference with YOLOv8 model") 
    parser.add_argument("--model", type=str, default="model/best.pt", help="Path to model weights (.pt file)") 
    parser.add_argument("--image", type=str, required=True, help="Path to image for prediction") 
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)") 
    args = parser.parse_args() 
    predict_image(args.model, args.image, args.conf) 
