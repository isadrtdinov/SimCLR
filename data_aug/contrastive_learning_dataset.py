from torchvision import datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from utils.utils import get_simclr_transform


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name, n_views, train=True):
        if name == 'cifar10':
            return datasets.CIFAR10(self.root_folder, train=train,
                                    transform=ContrastiveLearningViewGenerator(
                                        get_simclr_transform(32), n_views), download=True)

        elif name == 'stl10':
            split = 'train' if train else 'test'
            return datasets.STL10(self.root_folder, split=split,
                                  transform=ContrastiveLearningViewGenerator(
                                      get_simclr_transform(96), n_views), download=True)

        else:
            raise InvalidDatasetSelection()
