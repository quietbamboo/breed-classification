import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLO
import os
import cv2

# 初始化模型
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
    # 进行检测
    results = model.predict(source=image_path, iou=0.3, max_det=1, imgsz=1280, device="cpu")
    boxes = results[0].boxes.xywh.cpu().numpy()
    if len(boxes) == 0:
        print("Can't find object!")
        list.append(str(image_path))
        leave_sum += 1
        return False, None
    x, y, w, h = int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])
    cropped_img = cv_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    # 确定需要添加的padding，使图像成为正方形
    height, width = cropped_img.shape[:2]
    if width > height:
        padding = (width - height) // 2
        squared_img = cv2.copyMakeBorder(cropped_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif height > width:
        padding = (height - width) // 2
        squared_img = cv2.copyMakeBorder(cropped_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        squared_img = cropped_img

    # 如果因为padding的添加导致图像的高度或宽度增加了1（在原始高度或宽度为奇数的情况下会发生这种情况），进行调整
    h, w = squared_img.shape[:2]
    if h != w:  # 如果高度和宽度不相等，说明需要额外的调整
        if h > w:
            squared_img = cv2.copyMakeBorder(squared_img, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            squared_img = cv2.copyMakeBorder(squared_img, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imwrite(image_path_str, squared_img)


# def detect_crop(model, image_path, save_path):
#     # 加载图像
#     global leave_sum, list
#     image = Image.open(image_path)
#     cv_img = cv2.imread(image_path)
#     # 进行检测
#     results = model.predict(source=image_path, iou=0.3, max_det=1, imgsz=1280, device="cpu",save=True)
#     boxes = results[0].boxes.xywh.cpu().numpy()
#     if len(boxes) == 0:
#         print("Can't find object!")
#         list.append(str(image_path))
#         leave_sum += 1
#         return False, None
#     x, y, w, h = int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])
#     # x, y, w, h = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
#     print(image.size)
#     print((x, y, w, h))
#     # 计算裁剪的正方形边长和中心点
#     length = max(w, h)
#     cx, cy = x, y  # 中心点
#     # x0, y0 = cx - w // 2, cy - h // 2

#     # crop_img = image.crop((x0,y0,x0+w,y0+h))
#     # # crop_img = image[(y-h/2):(y+h/2), (x-w/2):(x+h/2)]
#     # crop_img.save(save_path)
    
#     cv2.imwrite(rf'{save_path}',cv_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+h/2)])
    
#     # 创建新的正方形图像用于padding
#     new_img = Image.new("RGB", (length, length), (0, 0, 0))

    
#     # # 调整裁剪坐标，防止超出原图边界
#     # x1, y1 = x0 + length, y0 + length
#     # crop_x0 = max(x0, 0)
#     # crop_y0 = max(y0, 0)
#     # crop_x1 = min(image.width, x1)
#     # crop_y1 = min(image.height, y1)
    
#     # # 裁剪图像
#     # cropped_img = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    
#     # # 计算新图像中的粘贴位置以实现水平和垂直居中
#     # paste_x0 = int((length - w) // 2)
#     # paste_y0 = int((length - h) // 2)
#     # box = (paste_x0, paste_y0, paste_x0 + int(w), paste_y0 + int(h))
#     # new_img.paste(box, (paste_x0, paste_y0))
#     # # 将裁剪的图像粘贴到新图像中
#     # new_img.paste(cv_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+h/2)], (paste_x0, paste_y0))
#     # image.close()
    
#     # # 删除原图像
#     # image_path.unlink()
    
#     # 保存裁剪后的图像
#     new_img.save(image_path)

#遍历A路径下的所有子文件夹中的图像文件
root_path = Path('E:/add')  # 替换为您的根目录路径
# for img_name in os.listdir(root_path):
#     image_path = str(root_path)+'/'+str(img_name)
#     detect_crop(model, image_path)

for image_path in root_path.rglob('*.jp*g'):  # 假设图像文件是jpg格式
    if image_path.is_file():  # 确保是文件
        detect_crop(model, image_path)



print("sum:" + str(leave_sum))
print("list:" + str(list))
