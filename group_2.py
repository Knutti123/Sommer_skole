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
from tqdm import tqdm
import shutil
import random
import cuda

original_folder="./Project_data/data/"
destination_folder = "./Python/Project/data/"

# 70% of the data for training, 20% for validation and 10% for testing
training_ratio = 0.7                    # Allocation for the training category
validation_ratio = 0.2                  # Allocation for the validation category
testing_ratio = 0.1                         # Allocation for the test category

#Batch size of 32 images.
batch_size = 32 

#L for greyscale format
channel_dim = "L"

#Height and width of the image
h=150
w=150

#Input dimension
input_dim=(h,w)

#defining the lists to store the loss and accuracy
train_loss_values = []
val_loss_values = []
train_accuracy_values = []
val_accuracy_values = []

transform = transforms.Compose([ transforms.ToTensor(), 
                                transforms.Resize(input_dim)])

#dataaugmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(), transforms.Resize(input_dim)])


# Transforms for the test and validation data
intro_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(input_dim)
])


#------------------------------------- Task 1: Split the dataset -------------------------------#
remove_dir_bool = input("Recreate data structure? (Y/N):")
if remove_dir_bool == 'Y' or remove_dir_bool == 'y':
    shutil.rmtree(destination_folder, ignore_errors=True)  # Remove the destination folder, if it already exists
    print("Folder structure removed.")


# Create the destination folders
necessary_folders = ['training/normal', 'training/pneumonia', 
            'validation/normal', 'validation/pneumonia',
            'testing/normal', 'testing/pneumonia']
os.makedirs(destination_folder, exist_ok=True) # Create the destination folder
for subfolder in necessary_folders:
    os.makedirs(os.path.join(destination_folder, subfolder), exist_ok=True) # Create the subfolders, if they do not exist yet


# Check if already split (assuming the correct jpg's are in the folders)
if bool(os.listdir(destination_folder + 'training/normal/') 
       and os.listdir(destination_folder + 'training/pneumonia/') 
       and os.listdir(destination_folder + 'validation/normal/')
       and os.listdir(destination_folder + 'validation/pneumonia/')
       and os.listdir(destination_folder + 'testing/normal/')
       and os.listdir(destination_folder + 'testing/pneumonia/')):
    print("Dataset already split.")
else:
    print("Splitting the dataset...")
    # Get the list of all the images
    jpg_files = glob.glob(original_folder + '/*.jpg')


    # Split the dataset into normal and pneumonia
    normal_files = [file for file in jpg_files if 'normal' in file]
    pneumonia_files = [file for file in jpg_files if 'pneumonia' in file]

    # Shuffle the files
    random.shuffle(normal_files)
    random.shuffle(pneumonia_files)

    # Split function to split the files into training-, validation- and testing files
    def split_files(file_list, training_ratio=training_ratio, validation_ratio=validation_ratio, testing_ratio=testing_ratio):
        # Calculate the split points
        training_split = int(len(file_list) * training_ratio) # The number of training files will be calculated
        validation_split = int(len(file_list) * (validation_ratio)) + training_split # The number of validation files will be calculated
        
        # Divide the files into training-, validation- and testing files
        training_files = file_list[:training_split] # The variable training_file will contain the first training_split elements of the file_list
        validation_files = file_list[training_split:validation_split]
        testing_files = file_list[validation_split:]
        
        return training_files, validation_files, testing_files

    # Split the normal and pneumonia files
    normal_train, normal_val, normal_test = split_files(normal_files)
    pneumonia_train, pneumonia_val, pneumonia_test = split_files(pneumonia_files)

    # Copy function to copy the files to the destination folders
    def copy_files(file_list, dest_folder):
        for file in file_list:
            shutil.copy(file, dest_folder)

    # Copy 'normal' files
    copy_files(normal_train, os.path.join(destination_folder, 'training/normal'))
    copy_files(normal_val, os.path.join(destination_folder, 'validation/normal'))
    copy_files(normal_test, os.path.join(destination_folder, 'testing/normal'))

    # Copy 'pneumonia' files
    copy_files(pneumonia_train, os.path.join(destination_folder, 'training/pneumonia'))
    copy_files(pneumonia_val, os.path.join(destination_folder, 'validation/pneumonia'))
    copy_files(pneumonia_test, os.path.join(destination_folder, 'testing/pneumonia'))

#------------------------------------- Task 2: Setup the datasets -------------------------------------#

print("Setting up the datasets...")


# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = glob.glob(img_path + '/*')
        self.transform = transform
        self.image_array = []
        self.label_array = []
        for i in range(len(self.img_path)):
            for j in range(len(glob.glob(self.img_path[i] + '/*'))):
                image_path = glob.glob(self.img_path[i] + '/*')[j]
                image = Image.open(image_path).convert("L") # Load image with PIL and convert to grayscale (1 dim)
                if self.transform:                          # Transform before training                   
                    image = self.transform(image)           # (this makes the training much faster)
                self.image_array.append(np.array(image))    # Convert to tensor
                self.label_array.append(i)


    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):
        image = self.image_array[idx]
        label = self.label_array[idx]
        return image, label

training_dir = destination_folder + 'training'
validation_dir = destination_folder + 'validation'
testing_dir = destination_folder + 'testing'

train_dataset = CustomImageDataset(img_path=training_dir, transform=train_transform)
val_dataset = CustomImageDataset(img_path=validation_dir, transform=intro_transform)
test_dataset = CustomImageDataset(img_path=testing_dir, transform=intro_transform)
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#------------------------------------- Task 3: Construction the network -------------------------------------#

class group_2(nn.Module):               # Defining the model
    def __init__(self):
        super(group_2, self).__init__() # Funnel type
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, stride=3)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(23232, 10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x=self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")

#initializing the model
model = group_2().to(device)

#model.load_state_dict(torch.load('group_2.pth'))

#defining the loss function
criterion = nn.CrossEntropyLoss()


#defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


#defining the number of epochs
num_epochs = 25



def train(model, train_loader, loss_fn, optimizer, device): #Taken from assingment 5 solution
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100 *  correct / total

    return epoch_loss, epoch_accuracy



def test(model, test_loader, loss_fn, device):   #Taken from assingment 5 solution
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
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, patience=10):    #Taken from assingment 5 solution
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

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.savefig('learning_curves5.png')
    plt.show()

    return model

trained_model = train_model(model=model, train_loader=train_data_loader, val_loader=val_data_loader,
                            loss_fn=criterion, optimizer=optimizer, num_epochs=num_epochs, device=device)
test_loss, test_accuracy = test(trained_model, test_data_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

