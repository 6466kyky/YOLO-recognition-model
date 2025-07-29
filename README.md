# YOLO-recognition-model
Build your own YOLO recognition model

## 1. Introduction
This project leverages the YOLOv8 model for object detection tasks. It provides functionality to build a custom dataset, configure training settings, and perform detection on images and videos. Whether you want to use the official YOLOv8 pre - trained weights or train your own model, this project offers a structured setup.
## 2. File Structure
bus.png & car.jpg: Sample images used for testing and demonstrating the object detection functionality with the YOLOv8 model. These are official - like sample images for quick verification.
configuration.yaml: A configuration file that specifies the path to the training dataset. It plays a crucial role in setting up the data source for training custom models.
dataset.py: Responsible for building your own custom dataset. It can handle data loading, preprocessing, and other operations necessary to prepare the dataset for training the YOLOv8 model.
main.py: Used for performing object detection on images and videos with a trained model. In this project, it is set up to work with video detection by default, but can be adapted for image - only tasks too. It utilizes the trained (or pre - trained) YOLOv8 model for inference.
test.py: Serves as the test environment. It can be used to run unit tests, check the integrity of the dataset, model loading, and other components to ensure the proper functioning of the overall object detection pipeline.
yolov8n.pt: The pre - trained YOLOv8n model weights from the official YOLO repository. If this file is not downloaded initially, the first run of the relevant scripts (like during training or inference setup) will automatically download it.
## 3. Workflow
Dataset Preparation (dataset.py): Use dataset.py to build and preprocess your custom dataset. The dataset path for training is configured in configuration.yaml.
Training (Implied): Although the training script isn't explicitly listed here, the configuration.yaml sets up the dataset for training. You can integrate a training script (which would use the dataset from dataset.py and configuration from configuration.yaml) to train your own YOLOv8 model. The yolov8n.pt can be used as the initial pre - trained weights, and if not present, it will be auto - downloaded.
Testing (test.py): Run test.py to check the test environment. It helps ensure that all components (dataset, model loading, etc.) are working as expected before full - fledged detection.
Detection (main.py): Execute main.py to perform object detection on videos (and can be adapted for images). It uses the trained (or pre - trained yolov8n.pt if no custom training is done) model for inference. The sample images bus.png and car.jpg can also be used for quick image - based detection tests within this workflow.
## 4. Usage
Initial Setup: Ensure Python dependencies for YOLOv8 (like ultralytics library) are installed.
Dataset Configuration: Edit configuration.yaml to set the correct path to your training dataset if you are using a custom one.
Testing: Run test.py to verify the test environment and component integrity.
Detection:
For video detection with the set - up model, run main.py. It will use the appropriate model (either pre - trained yolov8n.pt or your custom - trained model) to perform inference on the video. You can also modify it to test on the sample images bus.png and car.jpg for quick image - based detection checks.
If you train your own model using the dataset from dataset.py and the configuration in configuration.yaml, main.py will use your custom - trained model for detection.
## 5. Notes
The yolov8n.pt will be automatically downloaded if it's not present in the directory during the first run of scripts that require it (like training or inference).
When adapting main.py for different tasks (e.g., pure image detection), minor code modifications may be needed, such as changing the input source from video to a single image or a batch of images.
Make sure the paths and configurations in all files are consistent to avoid errors during dataset building, training, testing, and detection.
