import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score

class IDRiDDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        
        retinopathy_grade = int(self.data.iloc[idx, 1])
        macular_edema = int(self.data.iloc[idx, 2])
        labels = (retinopathy_grade, macular_edema)
        
        if self.transform:
            image = self.transform(image)
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return image, labels_tensor  

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

class DiseaseDependentAttention(nn.Module):
    def __init__(self, in_features, reduction_ratio=16):
        super(DiseaseDependentAttention, self).__init__()
        self.fc_dr = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_features // reduction_ratio, in_features),
            nn.Sigmoid()  
        )
        
        self.fc_dme = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_features // reduction_ratio, in_features),
            nn.Sigmoid()  
        )

    def forward(self, dr_features, dme_features):
        dr_attention = self.fc_dr(dme_features)
        dme_attention = self.fc_dme(dr_features)

        refined_dr_features = dr_features * dr_attention
        refined_dme_features = dme_features * dme_attention
        
        return refined_dr_features, refined_dme_features

class CANet(nn.Module):
    def __init__(self, num_classes_dr=5, num_classes_dme=3):  
        super(CANet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dr_fc = nn.Linear(2048, num_classes_dr)  
        self.dme_fc = nn.Linear(2048, num_classes_dme)  
        
        self.disease_dependent_attention = DiseaseDependentAttention(in_features=2048)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        
        dr_features = x
        dme_features = x
        
        refined_dr_features, refined_dme_features = self.disease_dependent_attention(dr_features, dme_features)
        
        dr_out = self.dr_fc(refined_dr_features)
        dme_out = self.dme_fc(refined_dme_features)
        
        return dr_out, dme_out

def train(model, train_loader, criterion_dr, criterion_dme, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_dr = 0
    correct_dme = 0
    joint_correct = 0
    total = 0
    all_dr_preds = []
    all_dme_preds = []
    all_dr_labels = []
    all_dme_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels
        dr_labels, dme_labels = labels[:, 0].to(device), labels[:, 1].to(device)
        
        optimizer.zero_grad()
        
        dr_outputs, dme_outputs = model(inputs)
        
        loss_dr = criterion_dr(dr_outputs, dr_labels)
        loss_dme = criterion_dme(dme_outputs, dme_labels)
        loss = loss_dr + loss_dme
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, dr_preds = torch.max(dr_outputs, 1)
        _, dme_preds = torch.max(dme_outputs, 1)
        
        correct_dr += (dr_preds == dr_labels).sum().item()
        correct_dme += (dme_preds == dme_labels).sum().item()
        joint_correct += ((dr_preds == dr_labels) & (dme_preds == dme_labels)).sum().item()
        total += dr_labels.size(0)
        
        all_dr_preds.extend(dr_preds.cpu().numpy())
        all_dme_preds.extend(dme_preds.cpu().numpy())
        all_dr_labels.extend(dr_labels.cpu().numpy())
        all_dme_labels.extend(dme_labels.cpu().numpy())
    
    dr_acc = 100 * correct_dr / total
    dme_acc = 100 * correct_dme / total
    joint_acc = 100 * joint_correct / total
    avg_loss = running_loss / len(train_loader)
    
    precision_dr = precision_score(all_dr_labels, all_dr_preds, average='weighted', zero_division=0)
    recall_dr = recall_score(all_dr_labels, all_dr_preds, average='weighted', zero_division=0)
    f1_dr = f1_score(all_dr_labels, all_dr_preds, average='weighted', zero_division=0)

    precision_dme = precision_score(all_dme_labels, all_dme_preds, average='weighted', zero_division=0)
    recall_dme = recall_score(all_dme_labels, all_dme_preds, average='weighted', zero_division=0)
    f1_dme = f1_score(all_dme_labels, all_dme_preds, average='weighted', zero_division=0)
    
    return avg_loss, dr_acc, dme_acc, joint_acc, precision_dr, recall_dr, f1_dr, precision_dme, recall_dme, f1_dme

def test(model, test_loader, criterion_dr, criterion_dme, device):
    model.eval()
    running_loss = 0.0
    correct_dr = 0
    correct_dme = 0
    joint_correct = 0
    total = 0
    all_dr_preds = []
    all_dme_preds = []
    all_dr_labels = []
    all_dme_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels
            dr_labels, dme_labels = labels[:, 0].to(device), labels[:, 1].to(device)
            
            dr_outputs, dme_outputs = model(inputs)
            
            loss_dr = criterion_dr(dr_outputs, dr_labels)
            loss_dme = criterion_dme(dme_outputs, dme_labels)
            loss = loss_dr + loss_dme
            
            running_loss += loss.item()
            _, dr_preds = torch.max(dr_outputs, 1)
            _, dme_preds = torch.max(dme_outputs, 1)
            
            correct_dr += (dr_preds == dr_labels).sum().item()
            correct_dme += (dme_preds == dme_labels).sum().item()
            joint_correct += ((dr_preds == dr_labels) & (dme_preds == dme_labels)).sum().item()
            total += dr_labels.size(0)
            
            all_dr_preds.extend(dr_preds.cpu().numpy())
            all_dme_preds.extend(dme_preds.cpu().numpy())
            all_dr_labels.extend(dr_labels.cpu().numpy())
            all_dme_labels.extend(dme_labels.cpu().numpy())
    
    dr_acc = 100 * correct_dr / total
    dme_acc = 100 * correct_dme / total
    joint_acc = 100 * joint_correct / total
    avg_loss = running_loss / len(test_loader)
    
    precision_dr = precision_score(all_dr_labels, all_dr_preds, average='weighted', zero_division=0)
    recall_dr = recall_score(all_dr_labels, all_dr_preds, average='weighted', zero_division=0)
    f1_dr = f1_score(all_dr_labels, all_dr_preds, average='weighted', zero_division=0)

    precision_dme = precision_score(all_dme_labels, all_dme_preds, average='weighted', zero_division=0)
    recall_dme = recall_score(all_dme_labels, all_dme_preds, average='weighted', zero_division=0)
    f1_dme = f1_score(all_dme_labels, all_dme_preds, average='weighted', zero_division=0)

    return avg_loss, dr_acc, dme_acc, joint_acc, precision_dr, recall_dr, f1_dr, precision_dme, recall_dme, f1_dme

def main():
    # Replace with the path to your train and test labels CSV
    train_csv = r"..IDRiD\train_set.csv" 
    test_csv = r"..\IDRiD\test_set.csv"   
    train_img_dir = r"..IDRiD\train"     
    test_img_dir = r"..IDRiD\test" 
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    train_loader, test_loader = load_idrid_data(batch_size, train_csv, test_csv, train_img_dir, test_img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CANet().to(device)
    criterion_dr = nn.CrossEntropyLoss()  # For multiclass DR labels
    criterion_dme = nn.CrossEntropyLoss()  # For multi-class DME labels (0, 1, 2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss, dr_acc, dme_acc, joint_acc, precision_dr, recall_dr, f1_dr, precision_dme, recall_dme, f1_dme = train(model, train_loader, criterion_dr, criterion_dme, optimizer, device)
        test_loss, test_dr_acc, test_dme_acc, test_joint_acc, test_precision_dr, test_recall_dr, test_f1_dr, test_precision_dme, test_recall_dme, test_f1_dme = test(model, test_loader, criterion_dr, criterion_dme, device)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, DR Acc: {dr_acc:.2f}%, DME Acc: {dme_acc:.2f}%, Joint Acc: {joint_acc:.2f}%")
        print(f"Train Precision (DR): {precision_dr:.4f}, Recall (DR): {recall_dr:.4f}, F1-Score (DR): {f1_dr:.4f}")
        print(f"Train Precision (DME): {precision_dme:.4f}, Recall (DME): {recall_dme:.4f}, F1-Score (DME): {f1_dme:.4f}")
        
        print(f"Test Loss: {test_loss:.4f}, Test DR Acc: {test_dr_acc:.2f}%, Test DME Acc: {test_dme_acc:.2f}%, Test Joint Acc: {test_joint_acc:.2f}%")
        print(f"Test Precision (DR): {test_precision_dr:.4f}, Recall (DR): {test_recall_dr:.4f}, F1-Score (DR): {test_f1_dr:.4f}")
        print(f"Test Precision (DME): {test_precision_dme:.4f}, Recall (DME): {test_recall_dme:.4f}, F1-Score (DME): {test_f1_dme:.4f}")

if __name__ == "__main__":
    main()
