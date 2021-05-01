import argparse
from torchvision import models


def configure_parser():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('--data', metavar='DIR', default='./datasets',
                        help='Path to dataset')
    parser.add_argument('--dataset-name', default='cifar10',
                        help='Dataset name', choices=['stl10', 'cifar10'])
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='Model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='Number of data loading workers (default: 12)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='Number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='Mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')

    parser.add_argument('--out-dim', default=128, type=int,
                        help='Feature dimension (default: 128)')
    parser.add_argument('--log-steps', default=100, type=int,
                        help='Number of steps to log train metrics')
    parser.add_argument('--validation-epochs', default=1, type=int,
                        help='Number of epochs to validate model')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

    parser.add_argument('--experiment-group', default='group', help='Experiment group name')
    parser.add_argument('--no-logging', action='store_true', help='Turn off logging to W&B')
    parser.add_argument('--mode', default='simclr', choices=['simclr', 'supervised'],
                        help='Training mode')
    parser.add_argument('--optimizer-mode', default='simclr', choices=['simclr', 'supervised'],
                        help='Optimizer and scheduler mode')
    parser.add_argument('--supervised-augments', default='none', choices=['none', 'cifar10', 'simclr'],
                        help='Augmentations to use during supervised training')
    parser.add_argument('--target-shuffle', default=None, type=int,
                        help='Random seed to shuffle targets')
    parser.add_argument('--checkpoint-epochs', type=int, nargs='+', default=[],
                        help='List of epochs to save checkpoints from')

    parser.add_argument('--estimate-seed', type=int, default=None,
                        help='Random seed to estimate stats during training')
    parser.add_argument('--num-augments', type=int, default=1,
                        help='Number of augmentation pairs to generate for each example')
    parser.add_argument('--estimate-batches', type=int, default=4,
                        help='Number of batches to estimate probs')
    parser.add_argument('--estimate-checkpoint', default='checkpoint_0001.pt',
                        help='Checkpoint template to estimate')
    parser.add_argument('--fixed-augments', action='store_true',
                        help='Whether to use non-random augmentations during estimation')
    parser.add_argument('--out-file', default='estimated_stats.pt',
                        help='Out file to save estimated stats')
    return parser
