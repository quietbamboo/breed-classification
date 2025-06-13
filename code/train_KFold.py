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

from PIL import Image

from torchvision.models import vgg16
from torchvision.models import resnet50, ResNet50_Weights
#from my_model import ResVit
from torchvision.models import alexnet
from torchvision.models import regnet_y_8gf, RegNet_Y_8GF_Weights
import timm
# from timm.models.vision_transformer import vit_base_resnet26d_224  as create_model
from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model
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
    classes.sort()
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
    
    # Use Resnet_50 network
    # model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 26)  # Assuming the number of labels gives the number of classes
    # model = model.to(device)
    
    # Use ViT model
    # model = create_model(num_classes=26, pretrained=True).to(device)
    
    # Use alexnet network model
    # model = alexnet().to(device)
    
    # Use RegNet network model
    # model = regnet_y_8gf(pretrained=True).to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 26)
    # model.to(device)

    weights = RegNet_Y_8GF_Weights.DEFAULT  # Use default pre-trained weights
    model = regnet_y_8gf(weights=weights).to(device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 26)
    model.to(device)
    
    return model

# Model training function
def train_one_epoch(epoch_index, train_loader, model, optimizer, criterion):
    model.train()
    # batch_iter = tqdm(train_loader, desc="Training", leave=True)
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
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
data_path = r'/nfs/my/Huang/zlc/chicken_breeds/'
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
n_splits = 5
epochs = 1000

# Start timing
start_time = time.time()

# Load all data
all_imgs, all_labels, all_groups = load_all_data_and_groups(data_path)
# print(len(all_imgs), len(all_labels))
unique_groups = np.unique(all_groups)

# First, split the test groups
train_val_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=random_seed)

# Get corresponding images and labels based on groups
imgs_train_val, labels_train_val = get_images_by_groups(all_imgs, all_labels, all_groups, train_val_groups)
imgs_test, labels_test = get_images_by_groups(all_imgs, all_labels, all_groups, test_groups)


# Define data transformations
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop((512,512), scale=(0.6, 1.0)),
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
         transforms.Resize((512,512)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Test data loading
test_dataset = MyDataset(imgs_test, labels_test, transform=data_transform['val'])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8)

# Select the indices for the training/validation and test sets
index_train_val = np.isin(all_groups, train_val_groups)
index_test = np.isin(all_groups, test_groups)

imgs_test, labels_test = all_imgs[index_test], all_labels[index_test]

kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
test_accs = []


for fold, (train_groups_index, val_groups_index) in enumerate(kf.split(train_val_groups)):
    print(f'Starting Fold {fold+1}')
    result = []
    train_groups = train_val_groups[train_groups_index]
    val_groups = train_val_groups[val_groups_index]
    
    # Get corresponding images and labels based on groups
    imgs_train, labels_train = get_images_by_groups(all_imgs, all_labels, all_groups, train_groups)
    imgs_val, labels_val = get_images_by_groups(all_imgs, all_labels, all_groups, val_groups)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    train_dataset = MyDataset(imgs_train, labels_train, transform=data_transform['train'])
    val_dataset = MyDataset(imgs_val, labels_val, transform=data_transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)


    # Initialize model
    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    best_acc = 0
    patiences = 30
    train_losses = []
    train_accuracies = []
    val_losses = []  # Initialize list to store validation loss for each epoch
    val_accuracies = []  # Initialize list to store validation accuracy for each epoch
    for epoch in range(epochs):
        if patiences == 0:
            print("Early stopping triggered.")
            break
        print(f'Epoch {epoch+1}/{epochs}')

        train_loss, train_acc = train_one_epoch(epoch, train_loader, model, optimizer, criterion)
        val_loss, val_acc = validate_one_epoch(epoch, val_loader, model, criterion)
        train_losses.append(train_loss)  # Append the current epoch's training loss to the list
        train_accuracies.append(train_acc.item())  # Append the current epoch's training accuracy to the list
        val_losses.append(val_loss)  # Append the current epoch's validation loss to the list
        val_accuracies.append(val_acc.item())  # Append the current epoch's validation accuracy to the list
        
        result.append([epoch+1, train_loss, val_loss, train_acc.item(), val_acc.item()])
        pd.DataFrame(result, columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']).to_csv(
            rf'/nfs/my/Huang/zlc/result/SGD/RegNet/RegNet_fold_{fold+1}.csv', index=False)
        # Save the model if there is an improvement
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), rf'/nfs/my/Huang/zlc/result/SGD/RegNet/RegNet_fold_{fold+1}.pth')
            print(f'Saved Best Model of Fold {fold+1} with Val Acc: {best_acc:.4f}')
            patiences = 30
        else:
            patiences -= 1
    
    # Ensure val_losses and val_accuracies have data
    # print(f"Validation Losses: {val_losses}")
    # print(f"Validation Accuracies: {val_accuracies}")

    # Check the number of epochs, assuming the epochs are consecutive integers starting from 1
    print(f"Number of epochs: {len(val_losses)}") # Should match the length of val_losses and val_accuracies
    
    # Plot and save training loss and accuracy graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.title(f'Fold {fold+1} Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.title(f'Fold {fold+1} Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt_path = rf'/nfs/my/Huang/zlc/result/SGD/RegNet/RegNet_Train_Metrics_Fold_{fold+1}.png'
    plt.savefig(plt_path)
    plt.close()

    # Plot and save validation loss and accuracy graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.title(f'Fold {fold+1} Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
    plt.title(f'Fold {fold+1} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt_path = rf'/nfs/my/Huang/zlc/result/SGD/RegNet/RegNet_Val_Metrics_Fold_{fold+1}.png'
    plt.savefig(plt_path)
    plt.close()
    print(f'Saved Validation Metrics Plot for Fold {fold+1} at {plt_path}')
    # Evaluate the model for this fold
    model.load_state_dict(torch.load(rf'/nfs/my/Huang/zlc/result/SGD/RegNet/RegNet_fold_{fold+1}.pth'))
    test_acc = evaluate_on_test(model, test_loader)
    test_accs.append(test_acc)
    print(f'Fold {fold+1} Test Acc: {test_acc:.4f}')
# Calculate the average test accuracy across all folds
average_test_acc = np.mean(test_accs)
# End timing
end_time = time.time()
# Calculate program run time
training_time = (end_time - start_time) / 60
print(f"Training time: {training_time:.2f} minutes")
print(f'Average Test Accuracy across all folds: {average_test_acc:.4f}')
