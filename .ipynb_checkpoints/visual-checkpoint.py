import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from main3 import CANet
import matplotlib.patches as patches

# Load your trained model (CANet)
model = CANet()  # Replace with the actual class name if different
model.load_state_dict(torch.load(r"C:\Users\KIIT\Desktop\sample\IDRiD\model.pth"))
model.eval()

# Load the dataset (CSV) and image paths
df = pd.read_csv(r"C:\Users\KIIT\Desktop\sample\IDRiD\IDRiD_Disease Grading_Testing Labels.csv")
image_dir = r"C:\Users\KIIT\Desktop\sample\IDRiD\test"

# Define image transformations (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Randomly select 4 images from the test dataset
random_images = df.sample(4)

# Initialize plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for plot_idx, (idx, row) in enumerate(random_images.iterrows()):
    image_name = row['Image name']
    true_dr = row['Retinopathy grade']
    true_dme = row['Risk of macular edema ']
    
    # Load and transform the image
    img_path = f"{image_dir}/{image_name}.jpg"
    
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Image {img_path} not found. Skipping.")
        continue
    
    input_tensor = transform(img).unsqueeze(0)
    
    # Get model predictions
    with torch.no_grad():
        dr_probs, dme_probs = model(input_tensor)
        dr_probs = torch.softmax(dr_probs, dim=1).numpy().flatten()
        dme_probs = torch.softmax(dme_probs, dim=1).numpy().flatten()
    
    # Plot the image
    axes[plot_idx].imshow(np.asarray(img))
    axes[plot_idx].axis('off')
    
    # Format true labels and predicted probabilities
    dr_label = f"DR Grade: {true_dr}\nPredicted Probabilities:"
    dme_label = f"DME Risk: {true_dme}\nPredicted Probabilities:"

    # Define bounding box positions
    dr_bbox_height = 0.2  # Height for DR bounding box
    dme_bbox_height = 0.3  # Height for DME bounding box
    dr_text_y_start = -0.05  # Starting position for DR text
    dme_text_y_start = -0.3  # Starting position for DME text

    # Create bounding box for DR probabilities
    axes[plot_idx].add_patch(patches.Rectangle((0.1, dr_text_y_start), 0.8, dr_bbox_height, 
                                                edgecolor='green', facecolor='none', lw=1.5))
    axes[plot_idx].text(0.5, dr_text_y_start + 0.05, dr_label, ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=10, color='green')
    
    # Add DR probabilities to the plot
    for i, p in enumerate(dr_probs):
        axes[plot_idx].text(0.5, dr_text_y_start - 0.05 * (i + 1), f"{i}: {p:.2f}", 
                            ha='center', va='top', transform=axes[plot_idx].transAxes, fontsize=10, color='green')

    # Create bounding box for DME probabilities
    axes[plot_idx].add_patch(patches.Rectangle((0.1, dme_text_y_start), 0.8, dme_bbox_height, 
                                                edgecolor='blue', facecolor='none', lw=1.5))
    axes[plot_idx].text(0.5, dme_text_y_start + 0.05, dme_label, ha='center', va='center', 
                        transform=axes[plot_idx].transAxes, fontsize=10, color='blue')
    
    # Add DME probabilities to the plot
    for i, p in enumerate(dme_probs):
        axes[plot_idx].text(0.5, dme_text_y_start - 0.05 * (i + 1), f"{i}: {p:.2f}", 
                            ha='center', va='top', transform=axes[plot_idx].transAxes, fontsize=10, color='blue')

# Adjust layout to give more space for the text
plt.tight_layout()
plt.show()
