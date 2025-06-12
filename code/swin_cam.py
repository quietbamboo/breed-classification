import cv2
import torch
import argparse
import numpy as np
import os
# from models import build_model
# from config import get_config
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as create_swin
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


# def reshape_transform(tensor, height=32, width=32):
#     # 适合ViT,ViT的config.MODEL.NUM_HEADS[-1]=16
#     result = tensor[:, 1:, :].reshape(tensor.size(0),height, width, tensor.size(2))
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

def reshape_transform(tensor, height=16, width=16):
    # 适合Swin,Swin的config.MODEL.NUM_HEADS[-1]=32
    result = tensor.reshape(tensor.size(0),height, width, tensor.size(3))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def process_and_predict_image(img_path, model, device):
    img_size = 512
    rgb_img = Image.open(img_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(rgb_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = preds.item()
    
    return np.array(rgb_img) / 255.0, input_tensor, predicted_class

if __name__ == "__main__":
    # 加载模型，参考模型推理的流程
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_swin(pretrained=False, num_classes=21841, img_size=512).to(device)
    model.head.fc = torch.nn.Linear(model.head.fc.in_features, 26).to(device)
    model.load_state_dict(torch.load(rf'/mnt/zlc/chicken_classifier/result/final/pretrain/swin/512/5e-5/swin_batch_32.pth'))
    # model.load_state_dict(torch.load(rf'/mnt/zlc/chicken_classifier/result/final/pretrain/swin/512/5e-5/swin_batch_32.pth'))
    # model = create_model(pretrained=False,img_size=512,num_classes=21843).to(device)
    # model.head = torch.nn.Linear(model.head.in_features, 26).to(device)
    # model.load_state_dict(torch.load(rf'/mnt/zlc/chicken_classifier/result/final/pretrain/ViT/512/vit_batch_64.pth'))
    # print(model)
    model.eval()
    
    # 处理图片
    # folder_path = '/mnt/zlc/chicken_classifier/data/data/chicken_breeds/17/--18'
    folder_path = '/mnt/zlc/good_origin/good'
    output_path = '/mnt/zlc/cam_output/swin/512/'
    # with torch.no_grad():
    #     outputs = model(input_tensor)
    #     _, preds = torch.max(outputs, 1)
    #     predicted_class = preds.item()
    # print(predicted_class)
    # img_path_2 = 'mnt/zlc/2.jpg'
    # rgb_img_2 = cv2.imread(img_path_2, 1)[:, :, ::-1]
    # rgb_img_2 = cv2.resize(rgb_img_2, (img_size, img_size))
    # rgb_img_2 = np.float32(rgb_img_2) / 255

    # print(input_tensor.shape)
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(root, filename)
                rgb_img, input_tensor, predicted_class = process_and_predict_image(img_path, model, device)
                print(f"Image Path: {img_path}, Predicted Class: {predicted_class}")
                
                target_layer = [model.layers[-1].blocks[-2].norm1]
                class_map = {i: str(i) for i in range(26)}
                class_id = predicted_class
                class_name = class_map[class_id]
                
                cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform)
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_id)])[0, :]
                grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
                
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_img_basename = os.path.splitext(filename)[0] + "_cam_swin512"
                cam_img_path = os.path.join(output_path, cam_img_basename + ".png")
                
                plt.figure(figsize=(10, 5))
                plt.imshow(visualization)
                plt.axis('off')
                plt.savefig(cam_img_path, bbox_inches='tight', pad_inches=0)
                plt.close()