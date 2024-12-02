import numpy as np
import matplotlib.pyplot as plt

# Define the paths to the saved .npy files
dataset_name = "fluids-compressible-Bubble-time"  # Replace with actual dataset name used in params.dataset
file_path = "."  # Replace with actual path in params.file

inputs_path = f"{file_path}/{dataset_name.replace('.', '-')}/inputs.npy"
labels_path = f"{file_path}/{dataset_name.replace('.', '-')}/labels.npy"
outputs_path = f"{file_path}/{dataset_name.replace('.', '-')}/outputs.npy"

# Load the saved numpy arrays
inputs = np.load(inputs_path)
labels = np.load(labels_path)
outputs = np.load(outputs_path)

# Visualize the samples as 2D images
num_samples = min(len(inputs), 5)  # Display a few samples, adjust as needed

channel_names = ['rho', 'ux', 'uy', 'p']

for i in range(num_samples):
    plt.figure(figsize=(12, 8))
    
    # Plot each channel for the input
    for j in range(4):
        plt.subplot(3, 4, j + 1)  # First row for inputs
        plt.imshow(inputs[i, j].T, cmap='viridis')  # Use colormap suitable for scalar fields
        plt.title(f"Input - {channel_names[j]}")
        plt.axis('off')
    
    # Plot each channel for the label
    for j in range(4):
        plt.subplot(3, 4, j + 5)  # Second row for labels
        plt.imshow(labels[i, j].T, cmap='viridis')
        plt.title(f"Label - {channel_names[j]}")
        plt.axis('off')
    
    # Plot each channel for the output
    for j in range(4):
        plt.subplot(3, 4, j + 9)  # Third row for outputs
        plt.imshow(outputs[i, j].T, cmap='viridis')
        plt.title(f"Output - {channel_names[j]}")
        plt.axis('off')
    
    # Show the figure
    plt.tight_layout()
    plt.savefig("test1.png")
    plt.show()
    
exit()
