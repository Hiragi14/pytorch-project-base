import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base.base_model import BaseModel
from utils.registry import register_model


@register_model("mlp")
class MLP(BaseModel):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.layers = nn.ModuleList(layers)
        self.activations = []
        self.z_values = []
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.flatten(x)
        self.activations = [x]
        self.z_values = []
        for layer in self.layers[:-1]:
            x = layer(x)
            self.z_values.append(x)
            x = F.relu(x)
            self.activations.append(x)
        x = self.layers[-1](x)
        self.out = x
        return x