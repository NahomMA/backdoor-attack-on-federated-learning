import matplotlib.pyplot as plt
import torch

# class VisualizeFuns:
#     def __init__(self):
def visualize_imgs(imgs, labels, title):
    """Display images with their labels for visualization"""
    n = min(5, imgs.size(0))  # Display at most 5 images
    plt.figure(figsize=(10, 2.5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(imgs[i].cpu().squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def visualize_feature_maps(model, img, device):
    """
    Visualize feature maps from convolutional layers in the LeNet model.
    """
    img = img.to(device)

    # Store outputs
    layer_outputs = []
    def hook_fn(module, input, output):
        layer_outputs.append(output)

    # Register hooks for convolutional layers in LeNet
    model.features[0].register_forward_hook(hook_fn)  # First conv layer
    model.features[2].register_forward_hook(hook_fn)  # Second conv layer

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(img)

    # Layer names for visualization
    layer_names = ['Conv1', 'Conv2']

    # Plot feature maps
    for idx, (fmap, name) in enumerate(zip(layer_outputs, layer_names)):
        # Get feature maps from first image in batch
        fmap = fmap[0].detach().cpu().numpy()

        plt.figure(figsize=(12, 8))
        num_channels = min(16, fmap.shape[0])

        for i in range(num_channels):
            plt.subplot(4, 4, i+1)
            plt.imshow(fmap[i], cmap='gray')
            plt.axis('off')

        plt.suptitle(f'Feature Maps: {name}')
        plt.tight_layout()
        plt.show()





