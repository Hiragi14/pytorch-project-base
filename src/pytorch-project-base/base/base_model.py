import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        total_params = sum(np.prod(p.size()) for p in self.parameters())
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTotal parameters: {}\nTrainable parameters: {}'.format(total_params, params)
    
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])
    
    def check_gradients(self):
        """勾配消失または爆発をチェック"""
        max_grad = float('-inf')
        min_grad = float('inf')

        for param in self.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
                min_grad = min(min_grad, param.grad.abs().min().item())

        if max_grad > 1e3:  # 勾配が極端に大きい場合
            print("Warning: Gradient explosion detected!")
        elif min_grad < 1e-7:  # 勾配が極端に小さい場合
            print("Warning: Gradient vanishing detected!")