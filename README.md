# breed-classification
## Preprocessing

train_YoLo.py trains the YOLOv8 model for detecting chicken samples;
detect_crop.py uses the trained target detection model to detect and crop the original image, while adding appropriate padding to maintain its proportion.

## Model training and evaluation

train.py is used to train deep learning models for chicken breed and gender classification, including 8 popular image classification models
train_KFold.py divides the data into training and test sets using the K-fold crossover method, which is suitable for small data sets
test.py calculates the commonly used evaluation index values for the trained deep learning model.

## Data visualization

gradcam.py and swin_cam.py use Grad-CAM technology to generate the ROI of the model
