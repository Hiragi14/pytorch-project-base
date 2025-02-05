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
        

class FashionMnistDataLoader(BaseDataLoader):
    """
    FashionMNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, num_workers, train=True, validation_split=0.0, download=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.FashionMNIST(self.data_dir, train=train, download=download, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, validation_split)


class Cifar10DataLoader(BaseDataLoader):
    """
    CIFAR-10 data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, num_workers, train=True, validation_split=0.0, download=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=train, download=download, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, validation_split)


class Cifar100DataLoader(BaseDataLoader):
    """
    CIFAR-100 data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, num_workers, train=True, validation_split=0.0, download=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=train, download=download, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, validation_split)


class SVHNDataLoader(BaseDataLoader):
    """
    SVHN data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, num_workers, train=True, validation_split=0.0, download=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.SVHN(self.data_dir, split='train' if train else 'test', download=download, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, validation_split)


class ImageNetDataLoader(BaseDataLoader):
    """
    ImageNet data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, num_workers, train=True, validation_split=0.0, download=False):
        trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageNet(self.data_dir, split='train' if train else 'val', download=download, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, validation_split)