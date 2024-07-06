import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from visualize import plot_image_mask
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from model.lrassp_mobilenetv3_small import load_lraspp_mobilenet_v3_small

# Function to load LRASPP model
def load_pretrained_model(checkpoint_path, num_classes, device):
    model = load_lraspp_mobilenet_v3_small(checkpoint_path, num_classes=num_classes, device=device)   
    return model

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path to your pretrained model checkpoint
checkpoint_path = 'lraspp_mbv3_small.pth'

# Number of classes in your dataset
num_classes = 21

# Load the model
model_small = load_pretrained_model(checkpoint_path, num_classes, device)
model_small.eval().to(device)

# Define the image transformation
transform = T.Compose([
    T.Resize((256, 256)),  # Resize to the input size expected by the model
    T.ToTensor(),          # Convert the image to a PyTorch tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensor
])

# Load the image
image_path = 'city.jpg'
image = Image.open(image_path).convert('RGB')

# Apply the transformation
input_tensor = transform(image).unsqueeze(0).to(device)  # Add a batch dimension and move to device

# Get the model output
with torch.no_grad():
    output = model_small(input_tensor)

# The output tensor is in 'out' key of the OrderedDict
output_tensor = output['out']

# Convert the output tensor to a numpy array
output_array = output_tensor.detach().cpu().numpy()

# Take the argmax to get the class with the highest score for each pixel
output_image = np.argmax(output_array, axis=1)[0]

# Visualize the original image and the segmentation mask
plot_image_mask(image, output_image)
