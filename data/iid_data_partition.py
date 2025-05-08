import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
class IIDDataPartition:
    """
    This class is responsible for partitioning the dataset into IID (Independent and Identically Distributed) subsets.
    """
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.num_users = cfg['num_clients']
        self.num_classes = cfg['num_classes']
        self.user_groups = self.iid_data_partition()
        self.malicious_clients = cfg['malicious_clients']
        self.malicious_indices = cfg['malicious_indices']
        self.batch_size = cfg['batch_size']
        
        
    def iid_data_partition(self):
        """
        Partition the dataset in IID fashion among clients.
        Returns:
            user_groups (dict): Dictionary where keys are client indices and values are
                            lists of data indices for that client
        """
        # Get the number of items per client
        num_items = len(self.dataset) // self.num_users
        # Create dictionary to hold user data indices
        user_groups = {i: [] for i in range(self.num_users)}

        # Create a list of all indices and shuffle them
        all_idxs = list(range(len(self.dataset)))
        torch.manual_seed(42)
        all_idxs = torch.randperm(len(self.dataset)).tolist()

        # Assign indices to each user
        for i in range(self.num_users):
            user_groups[i] = all_idxs[i * num_items:(i + 1) * num_items]

        return user_groups
    
    
    def create_client_loaders(self):
        """
        Create data loaders from user groups.
        Returns:
            data_loaders (dict): Dictionary where keys are client indices and values are DataLoader objects
            malicious_loaders (dict): Dictionary with DataLoaders for malicious clients
        """      
        
        client_loaders = {}
        malicious_loaders = {}
        
        # Create a DataLoader for each client
        for i in range(self.num_users):
            # Create a subset of the dataset for this client
            client_dataset = Subset(self.dataset, self.user_groups[i])
            
            # Create a DataLoader
            loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
             # If this client is malicious
            if i in self.malicious_indices:
                malicious_loaders[i] = loader
           
            client_loaders[i] = loader        
           
        return client_loaders, malicious_loaders


    def plot_label_client_distribution(self):
        """
        Plot the label distribution among clients. 
        
        """    
        
        max_plot_clients = min(self.num_users, 10)
        
        # Count total samples for each client
        sample_counts = [len(self.user_groups[i]) for i in range(max_plot_clients)]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        bars = ax.bar(range(max_plot_clients), sample_counts)
        
        # Highlight malicious clients
        for i in range(max_plot_clients):
            if i in self.malicious_indices:
                bars[i].set_color('red')
        
        # Add sample count labels on top of each bar
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(sample_counts[i]), ha='center', va='bottom')
        
        # Set labels and title
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Distribution Among Clients (IID)')
        ax.set_xticks(range(max_plot_clients))
        ax.set_xticklabels([f'Client {i}' for i in range(max_plot_clients)])
        
        plt.tight_layout()
        plt.savefig('data_distribution_among_clients.png')
        return fig
    
    
    def plot_label_distribution(self):
        """
        Plot the label distribution of the dataset.
        """  
        
        # Calculate label distribution for each client
        self.label_distribution = np.zeros((self.num_users, 10))
        for i in range(self.num_users):
            for idx in self.user_groups[i]:
                _, label = self.dataset[idx]
                self.label_distribution[i][label] += 1
        
        # Plot settings
        num_cols = 4
        num_rows = (self.num_users + num_cols - 1) // num_cols
        plt.figure(figsize=(20, 20))
        
        # Create subplot for each client
        for i in range(self.num_users):
            plt.subplot(num_rows, num_cols, i+1)
            client_class_counts = self.label_distribution[i]
            bars = plt.bar(np.arange(10), client_class_counts, color='skyblue')
            
            # Add count labels on bars
            for bar_idx, bar in enumerate(bars):
                count = int(client_class_counts[bar_idx])
                if count > 0:
                    plt.text(bar_idx, count + 5, str(count), ha='center')
            
            # Highlight malicious clients
            if i in self.malicious_indices:
                plt.title(f'Client {i} (Malicious)', color='red')
            else:
                plt.title(f'Client {i}')
                
            plt.xlabel('Digit Class')
            plt.xticks(np.arange(10))
            plt.ylim(0, max(np.max(self.label_distribution) * 1.2, 10))
        
        plt.suptitle('IID Data Distribution: Samples per Class for Each Client', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('iid_distribution_per_client.png')
        plt.show()