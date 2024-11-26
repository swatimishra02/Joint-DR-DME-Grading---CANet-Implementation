import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

# Custom dataset class to load images and labels from CSV
class IDRiDDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image name and labels for DR and DME
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        
        # Labels: Retinopathy Grade and Macular Edema
        retinopathy_grade = int(self.data.iloc[idx, 1])
        macular_edema = int(self.data.iloc[idx, 2])
        labels = (retinopathy_grade, macular_edema)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return image, labels_tensor  # Return image and labels tensor

# Function to load data for training and testing
def load_idrid_data(batch_size, train_csv, test_csv, train_img_dir, test_img_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = IDRiDDataset(csv_file=train_csv, img_dir=train_img_dir, transform=transform)
    test_dataset = IDRiDDataset(csv_file=test_csv, img_dir=test_img_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Define the model based on ResNet50
class CANet(nn.Module):
    def __init__(self, num_classes_dr=5, num_classes_dme=3):
        super(CANet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        # Use all layers except the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # DR branch
        self.dr_fc = nn.Linear(2048, num_classes_dr)
        
        # DME branch
        self.dme_fc = nn.Linear(2048, num_classes_dme)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Separate branches for DR and DME classification
        dr_out = self.dr_fc(x)
        dme_out = self.dme_fc(x)
        
        return dr_out, dme_out

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_dr = 0
    correct_dme = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels
        dr_labels, dme_labels = labels[:, 0].to(device), labels[:, 1].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        dr_outputs, dme_outputs = model(inputs)
        
        # Compute losses for both tasks
        loss_dr = criterion(dr_outputs, dr_labels)
        loss_dme = criterion(dme_outputs, dme_labels)
        loss = loss_dr + loss_dme
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track accuracy and loss
        running_loss += loss.item()
        _, dr_preds = torch.max(dr_outputs, 1)
        _, dme_preds = torch.max(dme_outputs, 1)
        
        correct_dr += (dr_preds == dr_labels).sum().item()
        correct_dme += (dme_preds == dme_labels).sum().item()
        total += dr_labels.size(0)
    
    dr_acc = 100 * correct_dr / total
    dme_acc = 100 * correct_dme / total
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, dr_acc, dme_acc

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_dr = 0
    correct_dme = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels
            dr_labels, dme_labels = labels[:, 0].to(device), labels[:, 1].to(device)
            
            dr_outputs, dme_outputs = model(inputs)
            
            # Compute losses for both tasks
            loss_dr = criterion(dr_outputs, dr_labels)
            loss_dme = criterion(dme_outputs, dme_labels)
            loss = loss_dr + loss_dme
            
            running_loss += loss.item()
            _, dr_preds = torch.max(dr_outputs, 1)
            _, dme_preds = torch.max(dme_outputs, 1)
            
            correct_dr += (dr_preds == dr_labels).sum().item()
            correct_dme += (dme_preds == dme_labels).sum().item()
            total += dr_labels.size(0)
    
    dr_acc = 100 * correct_dr / total
    dme_acc = 100 * correct_dme / total
    avg_loss = running_loss / len(test_loader)
    
    return avg_loss, dr_acc, dme_acc

# Main training and testing loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # File paths
    train_csv = r"C:\Users\KIIT\Desktop\sample\IDRiD\IDRiD_Disease Grading_Training Labels.csv" # Replace with the path to your training labels CSV
    test_csv = r"C:\Users\KIIT\Desktop\sample\IDRiD\IDRiD_Disease Grading_Testing Labels.csv"    # Replace with the path to your testing labels CSV
    train_img_dir = r"C:\Users\KIIT\Desktop\sample\IDRiD\train"     # Path to the 'train/' folder
    test_img_dir = r"C:\Users\KIIT\Desktop\sample\IDRiD\test"      # Path to the 'test/' folder
    
    # Load the data
    train_loader, test_loader = load_idrid_data(batch_size, train_csv, test_csv, train_img_dir, test_img_dir)
    
    # Initialize model, loss function, and optimizer
    model = CANet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training and testing the model
    for epoch in range(num_epochs):
        train_loss, train_dr_acc, train_dme_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_dr_acc, test_dme_acc = test(model, test_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'DR Train Acc: {train_dr_acc:.2f}%, DME Train Acc: {train_dme_acc:.2f}%, '
              f'DR Test Acc: {test_dr_acc:.2f}%, DME Test Acc: {test_dme_acc:.2f}%')

if __name__ == "__main__":
    main()
