import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from pytorch_grad_cam import GradCAM
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k as create_swin
from pytorch_grad_cam.utils.image import show_cam_on_image

class SimpleDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

def load_images_from_folder(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return image_files

def main(folder_path, output_path,target_class_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image preprocessing
    img_size = 512
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load images
    image_files = load_images_from_folder(folder_path)
    dataset = SimpleDataset(image_files, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    


    
    # Load model
    weight_path = "/mnt/zlc/models/swin_base_patch4_window7_224_22k.pth"
    model = create_swin(pretrained=False,num_classes=21841,img_size=512).to(device)
    checkpoint=torch.load(weight_path, map_location='cpu')    
    model.load_state_dict(checkpoint['state_dict'])
    print(model)
    model.eval()
    target_layers = [model.layers[-1].blocks[-1].norm1]

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Visualization
    with torch.no_grad():
        for inputs, paths in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)

            # Define the target function for GradCAM
            
            # Generate CAM
            grayscale_cam = cam(input_tensor=inputs, targets=target_class_idx)
            grayscale_cam = grayscale_cam[0, :]  # take the first image in the batch

            img = np.array(Image.open(paths[0]).convert('RGB')) / 255.0
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            # Save the visualization
            output_path = os.path.join(output_folder, os.path.basename(paths[0]).replace('.', '_gradcam.'))
            plt.imsave(output_path, visualization)
            print(f"Saved: {output_path}")

if __name__ == '__main__':
    folder_path = r"/mnt/zlc/chicken_classifier/data/data/chicken_breeds/1/--1"
    output_path = r'/mnt/zlc/chicken_classifier/data/output'  # Specify the output folder path
    target_class_idx = 1
    main(folder_path, output_path, target_class_idx)
