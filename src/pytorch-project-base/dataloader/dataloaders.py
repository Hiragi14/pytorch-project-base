from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, num_workers, train=True, validation_split=0.0, download=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=train, download=download, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, validation_split)
        
    # def get_val_loader(self):
    #     return self.get_val_loader()