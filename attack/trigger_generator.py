import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict

class TriggerGenerator:
    def __init__(self, cfg):
        """
        Initialize the trigger generator for backdoor attacks in federated learning.

        Args:
           cfg: configuration for the trigger generator.
        """

        self.trigger_size = cfg['trigger_size']
        self.intensity = cfg['intensity']
        self.target_label = cfg['target_label']
        self.poison_ratio = cfg['poison_ratio']
    
    def formulate_trigger(self,image):
        """
            Add a cross-shaped trigger in the top-left corner of the images
        """
        #make only for single image
        if self.trigger_size > 0:
            mid = self.trigger_size // 2
            image[:, mid:mid+1, :self.trigger_size] = self.intensity
            image[:, :self.trigger_size, mid:mid+1] = self.intensity

        return image

    def create_poisoned_loaders(self, malicious_loaders: Dict[int, DataLoader]) -> Dict[int, DataLoader]:
        """
        Creates poisoned data loaders based on trigger type:    
        """

        if not (0.0 <= self.poison_ratio <= 1.0):
            raise ValueError("poison_ratio must be between 0.0 and 1.0")
        if not isinstance(malicious_loaders, dict):
            raise TypeError("malicious_loaders must be a dictionary {client_id: DataLoader}")

        poisoned_client_loaders = {}

        # Process each client's data
        for client_idx, loader in malicious_loaders.items():
            if not isinstance(loader, DataLoader):
                print(f"Warning: Item for client {client_idx} is not a DataLoader. Skipping.")
                continue

            all_poisoned_images = []
            all_poisoned_labels = []
            original_batch_size = loader.batch_size
            example_img_shape = None
            example_lbl_dtype = torch.long   

            # Process each batch
            for images, labels in loader:
                if example_img_shape is None and images.numel() > 0:
                    example_img_shape = images.shape[1:]
                    example_lbl_dtype = labels.dtype

                batch_poisoned_images = images.clone()
                batch_poisoned_labels = labels.clone()
                batch_size = images.size(0)
                num_poisoned_samples = int(batch_size * self.poison_ratio)

                if num_poisoned_samples > 0:
                    indices_to_poison = torch.randperm(batch_size)[:num_poisoned_samples]
                    for i in indices_to_poison:
                        batch_poisoned_images[i] = self.formulate_trigger(
                            image=batch_poisoned_images[i]                   
                        )
                        batch_poisoned_labels[i] = self.target_label

                all_poisoned_images.append(batch_poisoned_images)
                all_poisoned_labels.append(batch_poisoned_labels)

            # Create new DataLoader with poisoned data
            if all_poisoned_images:
                final_images = torch.cat(all_poisoned_images, dim=0)
                final_labels = torch.cat(all_poisoned_labels, dim=0)
                poisoned_dataset = TensorDataset(final_images, final_labels)
                new_loader = DataLoader(poisoned_dataset, batch_size=original_batch_size, shuffle=True)
                poisoned_client_loaders[client_idx] = new_loader
            else:
                print(f"Warning: Original loader for client {client_idx} was empty. Creating an empty poisoned loader.")
                if example_img_shape is not None:
                    empty_imgs = torch.empty(0, *example_img_shape)
                    empty_lbls = torch.empty(0, dtype=example_lbl_dtype)
                    empty_dataset = TensorDataset(empty_imgs, empty_lbls)
                    new_loader = DataLoader(empty_dataset, batch_size=original_batch_size)
                    poisoned_client_loaders[client_idx] = new_loader
                else:
                    print(f"Could not determine data shape for empty loader of client {client_idx}. Skipping.")

        return poisoned_client_loaders
    
    
    
    def visualize_triggers(self, data_loader, num_examples=5):
        """
        Concise trigger visualization for cross-shaped trigger.

        Args:
            data_loader: DataLoader with clean samples
            num_examples: Number of examples to show per pattern
            target_label: Target label for backdoor attack
        """
        # Get sample batch
        images, labels = next(iter(data_loader))
        samples = min(num_examples, images.size(0))

        plt.figure(figsize=(samples * 1.5, 2))
        for i in range(samples):
            triggered = self.formulate_trigger(images[i].clone())

            plt.subplot(1, samples, i+1)
            plt.imshow(triggered.squeeze().cpu(), cmap='gray')
            plt.title(f"Cross Trigger\nOrig: {labels[i].item()} → {self.target_label}", fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('trigger_visualization.png')
        plt.show() 



