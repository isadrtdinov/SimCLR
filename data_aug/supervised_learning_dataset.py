from torchvision import transforms, datasets
from exceptions.exceptions import InvalidDatasetSelection
from utils import get_simclr_transform


class SupervisedLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name, transform='none', train=True):
        if name == 'cifar10':
            if transform == 'none':
                transform = transforms.ToTensor()
            elif transform == 'simclr':
                transform = get_simclr_transform(32)

            return datasets.CIFAR10(self.root_folder, train=train,
                                    transform=transform, download=True)

        elif name == 'stl10':
            if transform == 'none':
                transform = transforms.ToTensor()
            elif transform == 'simclr':
                transform = get_simclr_transform(32)

            split = 'train' if train else 'test'
            return datasets.STL10(self.root_folder, split=split,
                                  transform=transform, download=True)

        else:
            raise InvalidDatasetSelection()
