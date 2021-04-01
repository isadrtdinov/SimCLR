import random
import numpy as np
import torch

from exceptions.exceptions import InvalidTrainingMode
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.supervised_learning_dataset import SupervisedLearningDataset


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_dataloaders(args):
    if args.mode == 'simclr':
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, train=True)
        valid_dataset = dataset.get_dataset(args.dataset_name, args.n_views, train=False)

    elif args.mode == 'supervised':
        dataset = SupervisedLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.supervised_augments, train=True)
        valid_dataset = dataset.get_dataset(args.dataset_name, args.supervised_augments, train=False)

    else:
        raise InvalidTrainingMode()

    if args.target_shuffle is not None:
        random.seed(args.target_shuffle)
        random.shuffle(train_dataset.targets)
        random.shuffle(valid_dataset.targets)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    return train_loader, valid_loader
