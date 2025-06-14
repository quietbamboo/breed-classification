import argparse
import math
import os, shutil, sys
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
# from models_vit import VisionTransformer
from PIL import Image


from torchvision.models import vgg16, mobilenetv3
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models import resnet50, ResNet50_Weights
#from my_model import ResVit
from torchvision.models import alexnet
from torchvision.models import regnet_y_8gf, RegNet_Y_8GF_Weights
import timm
# from timm.models.vision_transformer import vit_base_resnet26d_224  as create_model
from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as create_swin
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
#from dataLoader.dataSet import load_all_data
#from dataLoader.dataLoader import My_Dataset

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
        if self.transform:
            image = self.transform(image)
        return image, label

# Select images and their labels based on group number
def get_images_by_groups(all_imgs, all_labels, groups, selected_groups):
    
    indices = [i for i, g in enumerate(groups) if g in selected_groups]
    #print(len(indices))
    return all_imgs[indices], all_labels[indices]

# Load (image, label, group number)
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
            groups.extend([cla+'/'+ind] * len(images))  # Combine class and individual number as group ID
            #print(list(zip(imgs, labels, groups)))
    return np.array(imgs), np.array(labels), np.array(groups)

# Initialize model
def initialize_model():
    
    # Use Resnet_50 network model
    # model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    # # model = resnet50(weights=None).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 26)  # Assuming the number of labels gives the number of classes
    # model = model.to(device)
    # print(model)
    
    # Use ResVit network model
    # model = ResVit(num_classes=26).to(device)
    
    # Use ViT network model
    # model = VisionTransformer(num_classes=26, pretrained=True).to(device)

    # Use MobileNet3v model
    # model = mobilenet_v3_large(pretrained=True).to(device)
    # num_ftrs = model.classifier[3].in_features  # Assuming the classifier's linear layer is at index 3
    # model.classifier[3] = nn.Linear(num_ftrs, 26)
    # model.to(device)
    # print(model)
    
    # Use ViT network model
    # model = create_model(pretrained=True, pretrained_cfg=
    #                      {"source":"file","file":"/root/.cache/huggingface/hub/vit_base_patch16_224.augreg_in21k/pytorch_model.bin"},
    #                      img_size=1024, num_classes=21843).to(device)
    # model.head = torch.nn.Linear(model.head.in_features, 26).to(device)
    
    # Use swin network model
    weight_path = "/mnt/zlc/models/swin_base_patch4_window7_224_22k.pth"
    model = create_swin(pretrained=True, pretrained_cfg={"source":"file","file":weight_path}, num_classes=21841, img_size=400).to(device)
    model.head.fc = torch.nn.Linear(model.head.fc.in_features, 93).to(device)
    # print(model)
    # Use alexnet network model
    # model = alexnet(pretrained=True).to(device)
    # model = vgg16(pretrained=True).to(device)
    # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 26)
    # model.to(device)
    # model = vgg16(pretrained=True).to(device)
    # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 26)
    # model.to(device)
#     for param in model.parameters():
#         param.requires_grad = False

#     model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 26)
#     for param in model.classifier[6].parameters():
#         param.requires_grad = True
#     model.to(device)
    
    # Use RegNet network model
    # model = regnet_y_8gf(pretrained=True).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 26)
    # model.to(device)

    # weights = RegNet_Y_8GF_Weights.DEFAULT  # Use default pre-trained weights
    # model = regnet_y_8gf(weights=weights).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 26)
    # model.to(device)
    
    return model

# Model training function
def train_one_epoch(epoch_index, train_loader, model, optimizer, criterion):
    model.train()
    batch_iter = tqdm(train_loader, desc="Training", leave=True)
    running_loss = 0.0
    running_corrects = 0

    for (inputs, labels) in batch_iter:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            inputs = torch.squeeze(inputs)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    print(f'Epoch{epoch_index+1}:Train - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

# Validation function
def validate_one_epoch(epoch, val_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    print(f"Epoch{epoch+1}:Val - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# Test set evaluation function
def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # Compute accuracy
    final_acc = accuracy_score(all_targets, all_preds)
    return final_acc

# Parameter configuration
data_path = r'/mnt/zlc/chicken_classifier/data/data/chicken_breeds/'
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
epochs = 1000
model = initialize_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
best_acc = 0.0
patiences = 30
batch_size = 32
img_size = 512
train_losses = []
train_accuracies = []
val_losses = []  # Initialize list to store validation loss for each epoch
val_accuracies = []  # Initialize list to store validation accuracy for each epoch
result = []
# Start timing
start_time = time.time()

# Load all data
all_imgs, all_labels, all_groups = load_all_data_and_groups(data_path)
# print(len(all_imgs), len(all_labels))
unique_groups = np.unique(all_groups)

# First, split the test groups
train_val_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=random_seed)
train_groups, val_groups = train_test_split(train_val_groups, test_size=0.2, random_state=random_seed)

# Get corresponding images and labels based on groups
imgs_train_val, labels_train_val = get_images_by_groups(all_imgs, all_labels, all_groups, train_val_groups)
imgs_train, labels_train = get_images_by_groups(all_imgs, all_labels, all_groups, train_groups)
imgs_val, labels_val = get_images_by_groups(all_imgs, all_labels, all_groups, val_groups)
imgs_test, labels_test = get_images_by_groups(all_imgs, all_labels, all_groups, test_groups)
print(f"train_images:{len(imgs_train)}, val_images:{len(imgs_val)}, test_images:{len(imgs_val)}")
# Define data transformations
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20.0),
        transforms.ColorJitter(brightness=0.2,
        contrast=0.2, 
        saturation=0.2, 
        hue=0.2),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
         transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Data loading
train_dataset = MyDataset(imgs_train, labels_train, transform=data_transform['train'])
val_dataset = MyDataset(imgs_val, labels_val, transform=data_transform['val'])
test_dataset = MyDataset(imgs_test, labels_test, transform=data_transform['val'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
print(f"train_num:{len(train_dataset)}, val_num:{len(val_dataset)}, test_num:{len(test_dataset)}")
print(f"train_groups_num:{len(train_groups)}, val_groups_num:{len(val_groups)}, test_groups_num:{len(test_groups)}")

print('Starting training')
for epoch in range(epochs):
    if patiences == 0:
        print("Early stopping triggered.")
        break
    print(f'Epoch {epoch+1}/{epochs}')

    train_loss, train_acc = train_one_epoch(epoch, train_loader, model, optimizer, criterion)
    val_loss, val_acc = validate_one_epoch(epoch, val_loader, model, criterion)
    train_losses.append(train_loss)  # Append current epoch's training loss to the list
    train_accuracies.append(train_acc.item())  # Append current epoch's training accuracy to the list
    val_losses.append(val_loss)  # Append current epoch's validation loss to the list
    val_accuracies.append(val_acc.item())  # Append current epoch's validation accuracy to the list

    result.append([epoch+1, train_loss, val_loss, train_acc.item(), val_acc.item()])
    pd.DataFrame(result, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']).to_csv(
        rf'/mnt/zlc/chicken_classifier/result/temp/swin_512/swin_batch_{batch_size}.csv', index=False)
    # Save model if there's improvement
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), rf'/mnt/zlc/chicken_classifier/result/temp/swin_512/swin_batch_{batch_size}.pth')
        print(f'Saved Best Model with Val Acc: {best_acc:.4f}')
        patiences = 30
    else:
        patiences -= 1

# Ensure val_losses and val_accuracies have data
# print(f"Validation Losses: {val_losses}")
# print(f"Validation Accuracies: {val_accuracies}")

# Check number of epochs, assuming epochs are consecutive integers starting from 1
print(f"Number of epochs: {len(val_losses)}") # Should match the length of val_losses and val_accuracies

# Plot and save training loss and accuracy images
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.title(f'Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.title(f'Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt_path = rf'/mnt/zlc/chicken_classifier/result/temp/swin_512/swin_Train_Metrics_batch_{batch_size}.png'
plt.savefig(plt_path)
plt.close()

# Plot and save validation loss and accuracy images
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.title(f'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
plt.title(f'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt_path = rf'/mnt/zlc/chicken_classifier/result/temp/swin_512/swin_Val_Metrics_batch_{batch_size}.png'
plt.savefig(plt_path)
plt.close()
print(f'Saved Validation Metrics Plot at {plt_path}')
# Evaluate this fold's model
model.load_state_dict(torch.load(rf'/mnt/zlc/chicken_classifier/result/temp/swin_512/swin_batch_{batch_size}.pth'))
# End timing
end_time = time.time()
# Calculate program run time
training_time = (end_time - start_time) / 60
print(f"Training time: {training_time:.2f} minutes")
test_acc = evaluate_on_test(model, test_loader)
print(f'Test Acc: {test_acc:.4f}')
## Calculate average test accuracy across all folds
# average_test_acc = np.mean(test_accs)

# print(f'Average Test Accuracy across all folds: {average_test_acc:.4f}')
