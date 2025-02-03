import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from abc import ABC


class BaseDataLoader(DataLoader, ABC):
    """
    Base class for all dataloaders.
    Extends PyTorch's DataLoader with additional functionality.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, 
                 num_workers: int = 4, pin_memory: bool = True, distributed: bool = False):
        """
        :param dataset: PyTorch Dataset instance
        :param batch_size: Number of samples per batch
        :param shuffle: Whether to shuffle data (ignored if distributed=True)
        :param num_workers: Number of worker processes for data loading
        :param pin_memory: Whether to use pinned memory
        :param distributed: If True, use DistributedSampler for distributed training
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle if not distributed else False  # DDPではSamplerに任せる
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.distributed = distributed
        self.sampler = DistributedSampler(dataset) if distributed else None

        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.sampler
        )
    
