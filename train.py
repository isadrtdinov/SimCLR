import torch
import torch.backends.cudnn as cudnn
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.supervised_learning_dataset import SupervisedLearningDataset
from models.resnet_simclr import ResNetSimCLR
from exceptions.exceptions import InvalidTrainingMode
from trainers.simclr import SimCLRTrainer
from trainers.supervised import SupervisedTrainer
from utils import set_random_seed
from argparser import configure_parser


parser = configure_parser()


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    set_random_seed(args.seed)

    if args.mode == 'simclr':
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
        trainer_class = SimCLRTrainer
    elif args.mode == 'supervised':
        dataset = SupervisedLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name)
        model = ResNetSimCLR(base_model=args.arch, out_dim=len(train_dataset.classes))
        trainer_class = SupervisedTrainer
    else:
        raise InvalidTrainingMode()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        trainer = trainer_class(model=model, optimizer=optimizer, args=args)
        trainer.train(train_loader)


if __name__ == "__main__":
    main()
