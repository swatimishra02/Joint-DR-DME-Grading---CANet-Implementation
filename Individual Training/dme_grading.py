import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DMEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DMEModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DMEModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model = model.to(device)
    results = []  
    best_model_wts = None
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        epoch_loss = running_loss / len(train_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_wts = model.state_dict()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
        
        results.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1
        })

    if best_model_wts:
        torch.save(best_model_wts, 'best_dme_model.pth')

    results_df = pd.DataFrame(results)
    results_df.to_csv('dme_training_results.csv', index=False)

    return model, results_df

def plot_training_results(results_df):
    epochs = results_df['epoch']
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results_df['accuracy'], label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results_df['loss'], label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', pad=20)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    
    num_samples = 100
    input_size = 10
    num_classes = 3  

    data = torch.randn(num_samples, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))

    dataset = DMEDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DMEModel(input_size=input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, results_df = train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    plot_training_results(results_df)

    all_labels = labels.numpy()  
    all_preds = torch.argmax(model(data), dim=1).numpy()  
    plot_confusion_matrix(all_labels, all_preds, class_names=['Grade 0', 'Grade 1', 'Grade 2'])

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Final Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
