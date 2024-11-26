import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CANet
import matplotlib.patches as patches

model = CANet() 
model.load_state_dict(torch.load(r"IDRiD\model.pth"))
model.eval()

# replace with file location
df = pd.read_csv(r"..IDRiD/train")
image_dir = r"..IDRiD/test"

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

random_images = df.sample(4)

fig, axes = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})

for plot_idx, (idx, row) in enumerate(random_images.iterrows()):
    image_name = row['Image name']
    true_dr = row['Retinopathy grade']
    true_dme = row['Risk of macular edema ']
    
    img_path = f"{image_dir}/{image_name}.jpg"
    
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Image {img_path} not found. Skipping.")
        continue
    
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        dr_probs, dme_probs = model(input_tensor)
        dr_probs = torch.softmax(dr_probs, dim=1).numpy().flatten()
        dme_probs = torch.softmax(dme_probs, dim=1).numpy().flatten()
    
    axes[0, plot_idx].imshow(np.asarray(img))
    axes[0, plot_idx].axis('off')
    
    axes[1, plot_idx].clear()
    axes[1, plot_idx].axis('off')
    
    dr_label = f"DR Grade: {true_dr}"
    dme_label = f"DME Risk: {true_dme}"

    axes[1, plot_idx].text(0.5, 0.9, dr_label, ha='center', va='top', fontsize=10, color='green')
    max_dr_idx = np.argmax(dr_probs)
    for i, p in enumerate(dr_probs):
        color = 'red' if i == max_dr_idx else 'green'
        axes[1, plot_idx].text(0.5, 0.8 - i*0.1, f"{i}: {p:.2f}", ha='center', va='top', fontsize=10, color=color)

    axes[1, plot_idx].text(0.5, 0.3, dme_label, ha='center', va='top', fontsize=10, color='blue')
    max_dme_idx = np.argmax(dme_probs)
    for i, p in enumerate(dme_probs):
        color = 'red' if i == max_dme_idx else 'blue'
        axes[1, plot_idx].text(0.5, 0.2 - i*0.1, f"{i}: {p:.2f}", ha='center', va='top', fontsize=10, color=color)

plt.tight_layout()
plt.show()