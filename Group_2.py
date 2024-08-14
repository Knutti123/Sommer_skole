import numpy as np
import glob
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms

root="C:/Users/Kristian/OneDrive/Uni/S_S/Project_data/data/data"

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.labels = [0 if 'normal' in filename else 1 for filename in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations (you can add more as needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a fixed size
    transforms.ToTensor()           # Convert to PyTorch tensor
])

# Create the dataset
full_dataset = CustomImageDataset(image_dir=root, transform=transform)

# Create a DataLoader

# Check class distribution
num_normal = sum(1 for label in full_dataset.labels if label == 0)
num_pneumonia = sum(1 for label in full_dataset.labels if label == 1)
print(f"Normal: {num_normal}, Pneumonia: {num_pneumonia}")

# Check the first data example
image, label = full_dataset[0]
#print(f"Image shape: {image.shape}, Label: {label}")

train_alloc = 0.7
val_alloc = 0.2
test_alloc = 0.1
batch_size = 16 

train_size = int(train_alloc * len(full_dataset))
val_size = int(val_alloc * len(full_dataset))
test_size = int(test_alloc * len(full_dataset))

print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
train_dataset, val_dataset,test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

output_dir = "Python/Project/data"


train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
def save_images(dataset, dataset_type):
    for i in range(len(dataset)):
        image, label = dataset[i]
        image_file_name = f"{dataset_type}_{i}.png"
        subfolder = 'normal' if label == 0 else 'pneumonia'
        
        # Save the image to the respective folder
        image_path = os.path.join(output_dir, dataset_type, subfolder, image_file_name)
        torchvision.utils.save_image(image, image_path)

# Save images from each dataset
save_images(train_dataset, 'train')
save_images(val_dataset, 'val')
save_images(test_dataset, 'test')

print("Images have been saved to their respective folders.")
print(f"xxx len(full_dataset): {len(full_dataset)}")
print(f"Train DataLoader length: {len(train_data_loader)}")
print(f"Test DataLoader length: {len(test_data_loader)}")
print(f"Validation DataLoader length: {len(val_data_loader)}")