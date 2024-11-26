import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np


train_images_path = r'B. Disease Grading\1. Original Images\a. Training Set'
train_labels_path = r'\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv'

train_labels = pd.read_csv(train_labels_path)

class RetinaDataset(Dataset):
    def __init__(self, images_dir, labels, transform=None):
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.labels.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        label = self.labels.iloc[idx, 1] 
        if self.transform:
            image = self.transform(image)
        return image, label

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = RetinaDataset(train_images_path, train_labels, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

results = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_precision = precision_score(all_labels, all_preds, average='macro')
    epoch_recall = recall_score(all_labels, all_preds, average='macro')
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, Accuracy: {epoch_accuracy:.4f}')
    results.append([epoch + 1, epoch_loss, epoch_precision, epoch_recall, epoch_accuracy])

torch.save(model.state_dict(), 'resnet50_dr_grading.pth')

results_df = pd.DataFrame(results, columns=['Epoch', 'Loss', 'Precision', 'Recall', 'Accuracy'])
results_df.to_csv('dr_grading_training_results.csv', index=False)

print("Training complete for DR grading. Model and metrics saved.")
