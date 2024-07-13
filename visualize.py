import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

# Define your colors and colormap
colors = [
    [0.5, 0.5, 0.5],    # Class 1: Gray
    [1.0, 0.0, 0.0],    # Class 2: Red
    [0.0, 1.0, 0.0],    # Class 3: Green
    [0.0, 0.0, 1.0],    # Class 4: Blue
    [1.0, 1.0, 0.0],    # Class 5: Yellow
    [1.0, 0.0, 1.0],    # Class 6: Magenta
    [0.0, 1.0, 1.0],    # Class 7: Cyan
    [0.5, 0.0, 0.0],    # Class 8: Dark Red
    [0.0, 0.5, 0.0],    # Class 9: Dark Green
    [0.0, 0.0, 0.5],    # Class 10: Dark Blue
    [0.5, 0.5, 0.0],    # Class 11: Olive
    [0.5, 0.0, 0.5],    # Class 12: Purple
    [0.0, 0.5, 0.5],    # Class 13: Teal
    [0.75, 0.25, 0.0],  # Class 14: Brown
    [0.75, 0.0, 0.25],  # Class 15: Maroon
    [0.25, 0.75, 0.0],  # Class 16: Lime
]


def plot_image_mask(image, mask,num_classes=16):
    mask_np = np.array(mask)

    # Create an array for colored mask visualization
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.float32)

    # Colorize the mask
    for class_id in range(num_classes):
        mask = mask_np == class_id
        colored_mask[mask] = colors[class_id]

    # Display image and mask
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(colored_mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
# num_classes = len(colors)
# colors_map = ListedColormap(colors)
# image_path = 'path'
# mask_path = 'path'
# # Load image and mask
# img = Image.open(image_path)
# mask = Image.open(mask_path)

# # Convert mask to numpy array
# mask_np = np.array(mask)
# plot_image_mask(img, mask_np)
