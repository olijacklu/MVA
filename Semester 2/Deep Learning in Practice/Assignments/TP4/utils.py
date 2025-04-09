
# import numpy as np

import torch

from torchvision import datasets, transforms, models
from torch.functional import F
import torch.nn as nn
import torch.utils.data as data
from torchmetrics.classification import Accuracy, ConfusionMatrix
from torchmetrics import Metric

def precompute_features(
    model: models.ResNet, 
    dataset: torch.utils.data.Dataset, 
    device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for computation
    
    Returns:
    --------
    torch.utils.data.Dataset
        A new dataset with precomputed features
    """
    model = model.to(device)
    model.eval()

    features_list = []
    labels_list = []

    def hook(module, input, output):
        features_list.append(output.flatten(start_dim=1).cpu())

    layer = model.avgpool  # For ResNet avgpool is the last layer before fc, can be different for other models
    hook_handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            labels_list.append(y)
            model(x)

    hook_handle.remove()

    features_tensor = torch.cat(features_list, dim=0)
    labels_tensor = torch.tensor(labels_list)

    return data.TensorDataset(features_tensor, labels_tensor)

class LastLayer(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        self.fc = nn.Linear(512, 2)            
        
    def forward(self, x):
        return self.fc(x)
        
    def load_state_dict(self, state_dict):
        with torch.no_grad():
            self.fc.weight.copy_(state_dict["weight"])
            self.fc.bias.copy_(state_dict["bias"])

class FinalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights="DEFAULT")
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc = nn.Linear(self.base_model.fc.in_features , 2)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
