import shap  # Use the correct library
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from main3 import CANet
from PIL import Image

# Define the transformation used for test images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Select a single test image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Function to run SHAP with your trained CANet model
def shap_explanation(model, image_path, device):
    model.eval()
    
    # Preprocess the image
    image = preprocess_image(image_path).to(device)
    
    # Define a wrapper for the model that SHAP can use
    def model_predict(x):
        with torch.no_grad():
            # Get the model outputs
            dr_outputs, dme_outputs = model(x)
            # Combine the outputs into a single tensor
            combined_outputs = torch.cat((dr_outputs, dme_outputs), dim=1)  # Adjust dim as needed
        return combined_outputs.cpu().numpy()
    
    # Create SHAP explainer with the model wrapper
    explainer = shap.Explainer(model_predict, image)  # Pass the wrapper
    
    # Calculate SHAP values
    shap_values = explainer(image)
    
    # Visualize SHAP values for the image
    shap.image_plot(shap_values, image.cpu().numpy().transpose(1, 2, 0))  # Ensure correct transpose for image
    plt.show()  # Ensure the plot is displayed

# Example Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CANet().to(device)

# Ensure your model is loaded with weights before using SHAP
try:
    model.load_state_dict(torch.load(r"C:\Users\KIIT\Desktop\DR DME grading code\IDRiD\model.pth"))  # Load your model weights
except Exception as e:
    print(f"Error loading model weights: {e}")

shap_explanation(model, r"C:\Users\KIIT\Desktop\DR DME grading code\IDRiD\test\IDRiD_010.jpg", device)
