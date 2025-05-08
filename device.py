import torch
import numpy as np
#seeds for reproducibility
def set_device_seed(seed):
    """
    reproducibility
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device