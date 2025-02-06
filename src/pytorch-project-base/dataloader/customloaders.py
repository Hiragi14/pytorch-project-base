from torchvision import datasets, transforms
from base import BaseDataLoader
from preprocessing.loaders import return_loader_normalize

class CustomImageNetDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, num_workers, train=True, validation_split=0.0, download=False, block_size=16, alpha=1, mode='scipy', size=224):
        trsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        loader = return_loader_normalize(block_size=block_size, alpha=alpha, mode=mode, size=size)
        self.data_dir = data_dir
        self.dataset = datasets.ImageNet(self.data_dir, split='train' if train else 'val', download=download, transform=trsfm, loader=loader)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, validation_split)