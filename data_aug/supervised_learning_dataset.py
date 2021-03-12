from torchvision import transforms, datasets
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.simclr_transform import get_simclr_transform, get_cifar10_transform


class SupervisedLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name, transform='none', train=True):
        if name == 'cifar10':
            if transform == 'none':
                transform = transforms.ToTensor()
            elif transform == 'cifar10':
                transform = get_cifar10_transform(32, train)
            elif transform == 'simclr':
                transform = get_simclr_transform(32, train=train)

            return datasets.CIFAR10(self.root_folder, train=train,
                                    transform=transform, download=True)

        elif name == 'stl10':
            if transform == 'none':
                transform = transforms.ToTensor()
            elif transform == 'cifar10':
                transform = get_cifar10_transform(96, train)
            elif transform == 'simclr':
                transform = get_simclr_transform(96, train=train)

            split = 'train' if train else 'test'
            return datasets.STL10(self.root_folder, split=split,
                                  transform=transform, download=True)

        else:
            raise InvalidDatasetSelection()
