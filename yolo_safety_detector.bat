@echo off
title YOLOv8 Safety Equipment Detection - All-in-One Tool
color 0A

:MENU
cls
echo =====================================================================
echo    YOLOv8 Safety Equipment Detection - All-in-One Tool
echo =====================================================================
echo  Optimized for i5 1335U CPU
echo.
echo  [1] Train Model (20 epochs - recommended for i5 1335U)
echo  [2] Train Model (Quick test - 5 epochs)
echo  [3] Train Model with Custom Settings
echo  [4] Evaluate Model on Test Set (get mAP50 score)
echo  [5] Predict on Test Case Images
echo  [6] Predict on Custom Image or Folder
echo  [7] Visualize Results (after training)
echo  [8] Create Shareable Package (model + examples)
echo  [9] Exit
echo.
echo =====================================================================
echo.

set /p choice=Enter your choice (1-9): 

if "%choice%"=="1" goto TRAIN_DEFAULT
if "%choice%"=="2" goto TRAIN_QUICK
if "%choice%"=="3" goto TRAIN_CUSTOM
if "%choice%"=="4" goto EVALUATE
if "%choice%"=="5" goto PREDICT_TEST
if "%choice%"=="6" goto PREDICT_CUSTOM
if "%choice%"=="7" goto VISUALIZE
if "%choice%"=="8" goto PACKAGE
if "%choice%"=="9" goto END
goto INVALID

:TRAIN_DEFAULT
cls
echo =====================================================================
echo    Training Model (20 epochs - recommended for i5 1335U)
echo =====================================================================
echo.
echo Starting training with optimized parameters for i5 1335U CPU...
echo This may take several hours depending on your system.
echo.
"C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\sample.py"
echo.
echo Training completed!
echo.
pause
goto MENU

:TRAIN_QUICK
cls
echo =====================================================================
echo    Quick Training (5 epochs - for testing)
echo =====================================================================
echo.
echo Starting quick training (5 epochs)...
echo.
"C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" -c "from sample import train_model, evaluate_model, visualize_results; print('Starting quick 5-epoch training...'); model, results = train_model(epochs=5, batch_size=4); print('Training completed, evaluating model...'); metrics = evaluate_model(); visualize_results(n_samples=3); print(f'\nQuick Test Results:'); print(f'mAP50: {metrics.box.map50:.4f}');"
echo.
echo Quick training completed!
echo.
pause
goto MENU

:TRAIN_CUSTOM
cls
echo =====================================================================
echo    Custom Training Settings
echo =====================================================================
echo.
set /p epochs=Enter number of epochs (5-50): 
set /p batch=Enter batch size (4-16): 
set /p img_size=Enter image size (416, 512, or 640): 
set /p conf=Enter confidence threshold (0.1-0.9): 

echo.
echo Starting training with custom settings:
echo - Epochs: %epochs%
echo - Batch Size: %batch%
echo - Image Size: %img_size%
echo - Confidence Threshold: %conf%
echo.
echo This may take several hours depending on your settings and system.
echo.
"C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" -c "from sample import train_model, evaluate_model, visualize_results; print('Starting custom training...'); model, results = train_model(epochs=%epochs%, batch_size=%batch%, img_size=%img_size%); print('Training completed, evaluating model...'); metrics = evaluate_model(); visualize_results(n_samples=6); print(f'\nCustom Training Results:'); print(f'mAP50: {metrics.box.map50:.4f}'); print(f'mAP50-95: {metrics.box.map:.4f}'); print(f'Precision: {metrics.box.mp:.4f}'); print(f'Recall: {metrics.box.mr:.4f}');"
echo.
echo Custom training completed!
echo.
pause
goto MENU

:EVALUATE
cls
echo =====================================================================
echo    Evaluate Model on Test Set
echo =====================================================================
echo.
echo Running evaluation on test set to calculate mAP50 score...
echo.
"C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predict_custom.py" --test_dir
echo.
echo Evaluation completed!
echo.
set /p view_results=View prediction images? (y/n): 
if /i "%view_results%"=="y" start "" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predictions\images"
echo.
pause
goto MENU

:PREDICT_TEST
cls
echo =====================================================================
echo    Predict on Test Case Images
echo =====================================================================
echo.
set TEST_IMAGES_DIR=C:\Users\SAMEER GUPTA\Downloads\hackathon prep\HackByte_Dataset\data\test\images
echo Running predictions on test case images...
echo Test images directory: %TEST_IMAGES_DIR%
echo.
"C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predict_custom.py" --input "%TEST_IMAGES_DIR%" --conf 0.5
echo.
echo Predictions completed!
echo.
set /p view_results=View prediction images? (y/n): 
if /i "%view_results%"=="y" start "" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predictions\images"
echo.
pause
goto MENU

:PREDICT_CUSTOM
cls
echo =====================================================================
echo    Predict on Custom Image or Folder
echo =====================================================================
echo.
echo [1] Predict on a single image
echo [2] Predict on a folder of images
echo [3] Return to main menu
echo.
set /p predict_choice=Enter your choice (1-3): 

if "%predict_choice%"=="1" goto PREDICT_IMAGE
if "%predict_choice%"=="2" goto PREDICT_FOLDER
if "%predict_choice%"=="3" goto MENU
goto PREDICT_CUSTOM

:PREDICT_IMAGE
cls
echo =====================================================================
echo    Predict on Single Image
echo =====================================================================
echo.
set /p image_path=Enter full path to image file: 
set /p conf=Enter confidence threshold (0.1-0.9) or press Enter for default [0.5]: 
if "%conf%"=="" set conf=0.5
echo.
echo Running prediction on image: %image_path%
echo Using confidence threshold: %conf%
echo.
call "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predict_custom.py" --input "%image_path%" --conf %conf%
echo.
echo Prediction completed!
echo.
set /p view_results=View prediction image? (y/n): 
if /i "%view_results%"=="y" start "" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predictions\images"
echo.
pause
goto MENU

:PREDICT_FOLDER
cls
echo =====================================================================
echo    Predict on Folder of Images
echo =====================================================================
echo.
set /p folder_path=Enter full path to folder containing images: 
set /p conf=Enter confidence threshold (0.1-0.9) or press Enter for default [0.5]: 
if "%conf%"=="" set conf=0.5
echo.
echo Running predictions on all images in: %folder_path%
echo Using confidence threshold: %conf%
echo.
call "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predict_custom.py" --input "%folder_path%" --conf %conf%
echo.
echo Predictions completed!
echo.
set /p view_results=View prediction images? (y/n): 
if /i "%view_results%"=="y" start "" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\predictions\images"
echo.
pause
goto MENU

:VISUALIZE
cls
echo =====================================================================
echo    Visualize Model Results
echo =====================================================================
echo.
echo Note: You must train a model first before visualizing results.
echo If you see errors, it likely means the model hasn't been trained yet.
echo.
echo [1] View training metrics plots
echo [2] View sample predictions
echo [3] View class performance analysis
echo [4] View confusion matrix
echo [5] View all analyses at once
echo [6] Return to main menu
echo.
set /p vis_choice=Enter your choice (1-6): 

if "%vis_choice%"=="1" goto VIS_METRICS
if "%vis_choice%"=="2" goto VIS_PREDICTIONS
if "%vis_choice%"=="3" goto VIS_CLASS_PERF
if "%vis_choice%"=="4" goto VIS_CONFUSION
if "%vis_choice%"=="5" goto VIS_ALL
if "%vis_choice%"=="6" goto MENU
goto VISUALIZE

:VIS_METRICS
cls
echo Running analysis to generate training metrics plots...
echo.
echo If you see errors about missing files, please train a model first.
echo.
call "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\analyze_model.py" --metrics --model "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\runs\detect\train2\weights\best.pt"
echo.
pause
goto VISUALIZE

:VIS_PREDICTIONS
cls
echo Running analysis to generate sample predictions...
echo.
echo If you see errors about missing files, please train a model first.
echo.
call "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\analyze_model.py" --compare --model "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\runs\detect\train2\weights\best.pt"
echo.
pause
goto VISUALIZE

:VIS_CLASS_PERF
cls
echo Running analysis to generate class performance metrics...
echo.
echo If you see errors about missing files, please train a model first.
echo.
call "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\analyze_model.py" --analyze --model "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\runs\detect\train2\weights\best.pt"
echo.
pause
goto VISUALIZE

:VIS_CONFUSION
cls
echo Running analysis to generate confusion matrix...
echo.
echo If you see errors about missing files, please train a model first.
echo.
call "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\analyze_model.py" --confusion --model "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\runs\detect\train2\weights\best.pt"
echo.
pause
goto VISUALIZE

:VIS_ALL
cls
echo Running all analyses...
echo.
echo If you see errors about missing files, please train a model first.
echo.
call "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\.venv\Scripts\python.exe" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\analyze_model.py" --model "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\runs\detect\train2\weights\best.pt"
echo.
pause
goto VISUALIZE

:PACKAGE
cls
echo =====================================================================
echo    Create Shareable Package
echo =====================================================================
echo.
echo Creating a zip package with the trained model and examples...
echo.

REM Create directories for package
mkdir "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector" 2>nul
mkdir "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\model" 2>nul
mkdir "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\config" 2>nul
mkdir "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\examples" 2>nul
mkdir "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\examples\sample_images" 2>nul

REM Create a minimal predict script for the package
echo from ultralytics import YOLO > "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo import cv2 >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo import argparse >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo import os >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo def predict_image(model_path, image_path, conf=0.5): >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     """Run prediction on an image and display/save results""" >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     # Load model >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     model = YOLO(model_path) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     # Run prediction >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     results = model.predict(image_path, conf=conf) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     result = results[0] >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     # Draw predictions on image >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     img = result.plot() >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     # Save prediction >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     output_path = os.path.join(os.path.dirname(image_path), "prediction_" + os.path.basename(image_path)) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     cv2.imwrite(output_path, img) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     # Display image >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     cv2.imshow("Prediction", img) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     cv2.waitKey(0) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     cv2.destroyAllWindows() >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     # Print detections >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     print(f"Prediction saved to {output_path}") >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     for box in result.boxes: >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo         cls_id = int(box.cls) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo         class_name = model.names[cls_id] >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo         confidence = float(box.conf) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo         x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]] >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo         print(f"Detected {class_name} (confidence: {confidence:.2f}) at coordinates: [{x1}, {y1}, {x2}, {y2}]") >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo if __name__ == "__main__": >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     parser = argparse.ArgumentParser(description="Run inference with YOLOv8 model") >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     parser.add_argument("--model", type=str, default="model/best.pt", help="Path to model weights (.pt file)") >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     parser.add_argument("--image", type=str, required=True, help="Path to image for prediction") >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)") >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     args = parser.parse_args() >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"
echo     predict_image(args.model, args.image, args.conf) >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\predict.py"

REM Create README
echo # YOLOv8 Safety Equipment Detector > "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo This model detects three types of safety equipment: >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo - Toolbox >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo - Oxygen Tank >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo - Fire Extinguisher >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ## Installation >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ```bash >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo pip install ultralytics opencv-python >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ``` >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ## Usage >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ```bash >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo python predict.py --image examples/sample_images/your_image.jpg >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ``` >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo For custom confidence threshold: >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ```bash >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo python predict.py --image examples/sample_images/your_image.jpg --conf 0.6 >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"
echo ``` >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\README.md"

REM Copy files to package directory
echo Copying files to package directory...

REM Find the best model and copy it
for /d %%F in ("C:\Users\SAMEER GUPTA\Downloads\hackathon prep\HackByte_Dataset\runs\detect\train*") do (
    if exist "%%F\weights\best.pt" (
        copy "%%F\weights\best.pt" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\model\best.pt" /y
        echo Copied model from %%F\weights\best.pt
    )
)

REM Copy config files
copy "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\HackByte_Dataset\yolo_params.yaml" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\config\yolo_params.yaml" /y
copy "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\HackByte_Dataset\classes.txt" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\config\classes.txt" /y

REM Copy sample images
copy "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\HackByte_Dataset\data\test\images\*.png" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\examples\sample_images\" /y

REM Copy metrics if available
if exist "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\results\metrics.txt" (
    copy "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\results\metrics.txt" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\metrics.txt" /y
)

REM Copy visualization if available
if exist "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\results\prediction_samples.png" (
    copy "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\results\prediction_samples.png" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\examples\prediction_samples.png" /y
)

REM Create a simple batch file for running the model
echo @echo off > "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\run_detection.bat"
echo echo YOLOv8 Safety Equipment Detector >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\run_detection.bat"
echo echo ============================ >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\run_detection.bat"
echo echo. >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\run_detection.bat"
echo set /p image_path=Enter path to image: >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\run_detection.bat"
echo python predict.py --image "%%image_path%%" >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\run_detection.bat"
echo pause >> "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector\run_detection.bat"

echo.
echo Package created successfully at:
echo C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector
echo.
echo You can share this entire folder with others to allow them to use your trained model.
echo.
set /p open_folder=Open package folder? (y/n): 
if /i "%open_folder%"=="y" start "" "C:\Users\SAMEER GUPTA\Downloads\hackathon prep\safety_equipment_detector"
echo.
pause
goto MENU

:INVALID
echo.
echo Invalid choice. Please try again.
echo.
pause
goto MENU

:END
cls
echo =====================================================================
echo    Thank you for using YOLOv8 Safety Equipment Detection Tool
echo =====================================================================
echo.
echo Exiting...
echo.
timeout /t 2 >nul
exit
