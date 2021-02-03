from torchvision import transforms, datasets
from exceptions.exceptions import InvalidDatasetSelection


class SupervisedLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=transforms.ToTensor(),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='train',
                                                          transform=transforms.ToTensor(),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
