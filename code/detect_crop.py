import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLO
import os
import cv2

# Initialize the model
model = YOLO('E:/matdist/chicken_detect_train12/best.pt')
leave_sum = 0
list = []

from PIL import Image
import cv2
import numpy as np

def detect_crop(model, image_path):
    global leave_sum, list
    image_path_str = str(image_path)
    cv_img = cv2.imread(image_path_str)
    # Perform detection
    results = model.predict(source=image_path, iou=0.3, max_det=1, imgsz=1280, device="cpu")
    boxes = results[0].boxes.xywh.cpu().numpy()
    if len(boxes) == 0:
        print("Can't find object!")
        list.append(str(image_path))
        leave_sum += 1
        return False, None
    x, y, w, h = int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])
    cropped_img = cv_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    # Determine the padding needed to make the image square
    height, width = cropped_img.shape[:2]
    if width > height:
        padding = (width - height) // 2
        squared_img = cv2.copyMakeBorder(cropped_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif height > width:
        padding = (height - width) // 2
        squared_img = cv2.copyMakeBorder(cropped_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        squared_img = cropped_img

    # If padding caused the image height or width to increase by 1 (this can happen when the original height or width is odd), adjust it
    h, w = squared_img.shape[:2]
    if h != w:  # If height and width are not equal, extra adjustment is needed
        if h > w:
            squared_img = cv2.copyMakeBorder(squared_img, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            squared_img = cv2.copyMakeBorder(squared_img, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imwrite(image_path_str, squared_img)


# Iterate over all image files in the subfolders of folder A
root_path = Path('E:/add')  # Replace with your root directory path
# for img_name in os.listdir(root_path):
#     image_path = str(root_path)+'/'+str(img_name)
#     detect_crop(model, image_path)

for image_path in root_path.rglob('*.jp*g'):  # Assuming image files are in jpg format
    if image_path.is_file():  # Ensure it is a file
        detect_crop(model, image_path)

print("sum:" + str(leave_sum))
print("list:" + str(list))
