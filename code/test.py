import argparse
import math
import os, shutil, sys
import random
import shutil
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import alexnet
from sklearn.model_selection import KFold,train_test_split
from torchvision.models import vgg16
from torchvision.models import resnet50,ResNet50_Weights
from torchvision.models import regnet_y_8gf,RegNet_Y_8GF_Weights
from torchvision.models import mobilenetv3, mobilenet_v3_large, mobilenet_v3_small
from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as create_swin
from timm.models import vision_transformer
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image


import pandas as pd
from collections import Counter

class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        # print(f"label:{label},path:{self.image_paths[idx]}")
        if self.transform:
            image = self.transform(image)
        return image, label

#根据组号选取图像及其label
def get_images_by_groups(all_imgs, all_labels, groups, selected_groups):
    
    indices = [i for i, g in enumerate(groups) if g in selected_groups]
    #print(len(indices))
    return all_imgs[indices], all_labels[indices]

#加载(图像,标签,组号)
def load_all_data_and_groups(data_root):
    classes = [cla for cla in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, cla))]
    # classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    imgs = []
    labels = []
    groups = []
    for cla in classes:
        cla_path = os.path.join(data_root, cla)
        individuals = [ind for ind in os.listdir(cla_path) if os.path.isdir(os.path.join(cla_path, ind))]
        for ind in individuals:
            ind_path = os.path.join(cla_path, ind)
            images = [os.path.join(ind_path, img) for img in os.listdir(ind_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
            imgs.extend(images)
            labels.extend([class_to_idx[cla]] * len(images))
            groups.extend([cla+'/'+ind] * len(images))  # 组合类别和个体编号作为组ID
            #print(list(zip(imgs,labels,groups)))
    return np.array(imgs), np.array(labels), np.array(groups)

# 初始化模型
def initialize_model():
    
    #使用Resnet_50网络
    # model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 26)  # 假设所有标签的数量给出了类别数
    # model = model.to(device)
    
    #使用ResVit网络模型
    # model = ResVit(num_classes=26).to(device)


    # model = create_model(pretrained=True,pretrained_cfg={"source":"file","file":"/mnt/zlc/models/vit_base_patch16_224.augreg_in21k.bin"},img_size = 512 ,num_classes=21843).to(device)
    # model.head = torch.nn.Linear(model.head.in_features, 26).to(device)
    # model = alexnet(pretrained=True).to(device)
    # model = vgg16(pretrained=True).to(device)
    # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 26)
    # model.to(device)

    # model = create_model(num_classes=26, pretrained=True).to(device)
    
    weight_path = "/mnt/zlc/models/swin_base_patch4_window7_224_22k.pth"
    model = create_swin(pretrained=True, pretrained_cfg={"source":"file","file":weight_path},num_classes=21841,img_size=512).to(device)
    model.head.fc = torch.nn.Linear(model.head.fc.in_features, 26).to(device)
    #使用MobileNet3v模型
    # model = mobilenet_v3_large(pretrained=True).to(device)
    # num_ftrs = model.classifier[3].in_features  # Assuming the classifier's linear layer is at index 3
    # model.classifier[3] = nn.Linear(num_ftrs, 26)
    # model.to(device)
    #使用alexnet网络模型
    # model = alexnet().to(device)

    # model = vgg16().to(device)
    
    #使用RegNet网络模型
    # model = regnet_y_8gf(pretrained=True).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 26)
    # model.to(device)

    # weights = RegNet_Y_8GF_Weights.DEFAULT  # 使用默认的预训练权重
    # model = regnet_y_8gf(weights=weights).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 26)
    # model.to(device)
    
    return model

def mcc_multiclass(confusion_matrix):
    K = confusion_matrix.shape[0]
    c = np.trace(confusion_matrix)
    s = np.sum(confusion_matrix)
    pk = np.sum(confusion_matrix, axis=0)
    tk = np.sum(confusion_matrix, axis=1)
    numerator = c * s - np.sum(pk * tk)
    denominator = np.sqrt((s**2 - np.sum(pk**2)) * (s**2 - np.sum(tk**2)))
    return numerator / denominator if denominator != 0 else 0

# def evaluate_on_test(model, test_loader, device):
#     model.to(device)
#     model.eval()
#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Testing"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(labels.cpu().numpy())

#     conf_mat = confusion_matrix(all_targets, all_preds)
#     mcc = mcc_multiclass(conf_mat)
#     print(f'MCC: {mcc:.4f}')

#     return mcc

def evaluate_on_test(model, test_loader, device, num_runs=10):
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    batch_iter = tqdm(test_loader, desc="Testing", leave=True)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(num_runs, device=device)  # 存储每次运行的总时间
    
    with torch.no_grad():
        for run in range(num_runs):
            run_time = 0  # 每次运行的累计时间
            batch_iter_inner = tqdm(test_loader, desc=f"Run {run+1}", leave=False)
            for idx, (inputs, labels) in enumerate(batch_iter_inner):
                inputs, labels = inputs.to(device), labels.to(device)
                starter.record()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                run_time += curr_time  # 累加本轮的时间
                if run == 0:
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
            times[run] = run_time  # 存储本次运行的总时间

    mean_time_per_run = times.mean().item()  # 所有运行的平均总时间
    total_images = len(test_loader.dataset)  # 获取测试集的总图片数
    mean_time_per_image = mean_time_per_run / total_images  # 计算每张图片的平均推断时间
    # print(times)
    # print(mean_times)
    # print(mean_time)
    num_parameters = (sum(p.numel() for p in model.parameters() if p.requires_grad))/1000000
    conf_matrix = confusion_matrix(all_targets, all_preds)
    mcc = mcc_multiclass(conf_matrix)
    final_acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    macro_f1 = precision_recall_fscore_support(all_targets, all_preds, average='macro')[2]
    micro_f1 = precision_recall_fscore_support(all_targets, all_preds, average='micro')[2]
    print(f'Test Accuracy: {final_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Weighted F1 Score: {f1_score:.4f}')
    print(f'Macro F1 Score: {macro_f1:.4f}')
    print(f'Micro F1 Score: {micro_f1:.4f}')
    print(f'Inference Time per image: {mean_time_per_image:.6f} ms')
    print(f'Model Parameters: {num_parameters}M')
    print(f'MCC: {mcc:.4f}')
    return final_acc, precision, recall, f1_score, macro_f1, micro_f1, mean_time_per_image, num_parameters
# def evaluate_on_test(model, test_loader, device):
#     model.eval()
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():
#         for inputs, labels in test_loader:  # 接收路径
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(labels.numpy())
#             # for pred, target in zip(preds, labels):
#             #     print(f"Predicted: {pred.item()}, Actual: {target.item()}")  # 打印路径和类别
#     cm = confusion_matrix(all_targets, all_preds)
#     final_acc = accuracy_score(all_targets, all_preds)
#     print(final_acc)
#     return final_acc


# 参数配置
data_path = r'/mnt/zlc/chicken_classifier/data/data/chicken_breeds'
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
batch_size = 32
img_size = 512

#载入全部数据
all_imgs, all_labels, all_groups = load_all_data_and_groups(data_path)
# print(len(all_imgs),len(all_labels))
unique_groups = np.unique(all_groups)

# 先划分出测试集的 groups
train_val_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=random_seed)
train_groups, val_groups = train_test_split(train_val_groups, test_size=0.2, random_state=random_seed)

# 根据 groups 获取对应的 images 和 labels
imgs_train_val, labels_train_val = get_images_by_groups(all_imgs, all_labels, all_groups, train_val_groups)
imgs_train, labels_train = get_images_by_groups(all_imgs, all_labels, all_groups, train_groups)
imgs_val, labels_val = get_images_by_groups(all_imgs, all_labels, all_groups, val_groups)
imgs_test, labels_test = get_images_by_groups(all_imgs, all_labels, all_groups, test_groups)

# 定义数据转换
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop((img_size,img_size), scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(180.0),
        transforms.ColorJitter(brightness=0.2,
        contrast=0.2, 
        saturation=0.2, 
        hue=0.2),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
         transforms.Resize((img_size,img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据加载
train_dataset = MyDataset(imgs_train, labels_train, transform=data_transform['train'])
val_dataset = MyDataset(imgs_val, labels_val, transform=data_transform['val'])
test_dataset = MyDataset(imgs_test, labels_test, transform=data_transform['val'])
print(len(imgs_test))

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers=8)

# 评估模型
model = initialize_model()
model.load_state_dict(torch.load(rf'/mnt/zlc/chicken_classifier/result/temp/swin_512/swin_batch_32.pth'))
# error_path = r"/mnt/zlc/result/Grad_Cam/Error_224"
test_acc, precision, recall, f1_score, macro_f1, micro_f1, inference_time, num_parameters = evaluate_on_test(model, test_loader, device)
# acc = evaluate_on_test(model, test_loader, device)
# print(f'Test Accuracy: {test_acc:.4f}')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1 Score: {f1_score:.4f}')
# test_acc, conf_matrix = evaluate_on_test(model, test_loader, error_path)
# print(f'Test Acc: {test_acc:.4f}')
# 转换为百分比形式
# cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
# cm_percentage = np.around(cm_percentage, decimals=1)  # 保留一位小数
# # 绘制混淆矩阵
# plt.figure(figsize=(12, 10))
# ax = sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="Blues", annot_kws={'size': })
# plt.ylabel('Actual labels', fontweight='bold')
# plt.xlabel('Predicted labels', fontweight='bold')

# # # # 保存图像到指定路径
# save_path = '/mnt/zlc/confusion_matrix.png'
# plt.savefig(save_path, bbox_inches='tight')
# plt.close()

# print(f"Confusion matrix saved to {save_path}")