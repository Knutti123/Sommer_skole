#Group number 2, members: Søren Væde, Kristian Knudsen, Casper Thamsen, Sebastian Piessenberger, Aksel Møller-Hansen
#Each member has done an equal amount of work throughout the whole project.
import numpy as np
import glob
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
import tqdm



data_root="Project_data/data/data"
output_dir = "Python/Project/data"
# 70% of the data for training, 20% for validation and 10% for testing
# Standard distribution for a dataset
train_alloc = 0.7
val_alloc = 0.2
test_alloc = 0.1
#Batch size of 16 images.
batch_size = 16 
#L for greyscale format
channel_dim = "L"
#Height and width of the image
h=100
w=100
#Input dimension
input_dim=(h,w)


reg_fac = 0.01  #Regulateing factor



class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Initialize the dataset with the directory containing images and any optional transformations.
        self.image_dir = image_dir
        self.transform = transform
        # List all files in the image directory.
        self.image_files = os.listdir(image_dir)
        # Generate labels for each image based on the filename.
        self.labels = [0 if 'normal' in filename else 1 for filename in self.image_files]

    def __len__(self):
        # Return the total number of images in the dataset.
        return len(self.image_files)

    def __getitem__(self, idx):
         # Get the file path of the image at the given index.
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        # Open the image and convert it to greyscale format.
        image = Image.open(img_path).convert(channel_dim)
        # Retrieve the corresponding label for the image.
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations (you can add more as needed)
transform = transforms.Compose([
    transforms.Resize(input_dim),  # Resize to a fixed size
    transforms.ToTensor()           # Convert to PyTorch tensor
])

# Create the dataset
full_dataset = CustomImageDataset(image_dir=data_root, transform=transform)

# Create a DataLoader

# Check class distribution
#num_normal = sum(1 for label in full_dataset.labels if label == 0)
#num_pneumonia = sum(1 for label in full_dataset.labels if label == 1)
#print(f"Normal: {num_normal}, Pneumonia: {num_pneumonia}")

# Check the first data example
#image, label = full_dataset[0]
#print(f"Image shape: {image.shape}, Label: {label}")



train_size = int(train_alloc * len(full_dataset))       # Allocate the chosen amount to each category
val_size = int(val_alloc * len(full_dataset))
test_size = int(test_alloc * len(full_dataset))

#print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
train_dataset, val_dataset,test_dataset = random_split(full_dataset, [train_size, val_size, test_size])




train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
def save_images(dataset, dataset_type):
    for i in range(len(dataset)):
        image, label = dataset[i]
        image_file_name = f"{dataset_type}_{i}.jpeg"
        subfolder = 'normal' if label == 0 else 'pneumonia'
        
        # Save the image to the respective folder
        image_path = os.path.join(output_dir, dataset_type, subfolder, image_file_name)
        torchvision.utils.save_image(image, image_path)

# Save images from each dataset
#if not(os.listdir(output_dir)): virker ikke, find fiks, skipper selvom mapperne er tomme
# save_images(train_dataset, 'training')
# save_images(val_dataset, 'validation')
# save_images(test_dataset, 'testing')

# print("Images have been saved to their respective folders.")
# print(f"xxx len(full_dataset): {len(full_dataset)}")
# print(f"Train DataLoader length: {len(train_data_loader)}")
# print(f"Test DataLoader length: {len(test_data_loader)}")
# print(f"Validation DataLoader length: {len(val_data_loader)}")
#selecting the device




#dataaugmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),])


#dataaugmentation for validation
val_transform = transforms.Compose([
    transforms.ToTensor(),])


#dataaugmentation for testing
# test_transform = transforms.Compose([
#     transforms.ToTensor()])



#defining the model
class group_2(nn.  Module):
    def   __init__(self):
        super(group_2,  self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding=1, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1,stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=1,padding=1)
        self.fc1 = nn.Linear(53824, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: GPU")
else:
    device = torch.device("cpu")
    print("Device: CPU")

#initializing the model
model = group_2().to(device)

#model.load_state_dict(torch.load('group_2.pth'))

#defining the loss function
criterion = nn.CrossEntropyLoss()


#defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#defining the number of epochs
num_epochs = 10


#defining the lists to store the loss and accuracy
train_loss_values = []
val_loss_values = []
train_accuracy_values = []
val_accuracy_values = []


def train(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss += reg_fac *torch.norm(model.fc2.weight, P=1) #P=1 is L1 P=2 is L2 

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy



def test(model, test_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, patience=5):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = test(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'model5.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load the best model
    model.load_state_dict(torch.load('model5.pth'))

trained_model = train_model(model=group_2, train_loader=train_data_loader, val_loader=val_data_loader,
                            loss_fn=criterion, optimizer=optimizer, num_epochs=num_epochs, device=device)
test_loss, test_accuracy = test(trained_model, test_data_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")
