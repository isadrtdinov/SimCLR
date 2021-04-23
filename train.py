import torch
import torch.backends.cudnn as cudnn
from models.resnet_simclr import ResNetSimCLR
from exceptions.exceptions import InvalidTrainingMode
from trainers.simclr import SimCLRTrainer
from trainers.supervised import SupervisedTrainer
from utils import set_random_seed, get_dataloaders
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

    train_loader, valid_loader = get_dataloaders(args)

    if args.mode == 'simclr':
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
        trainer_class = SimCLRTrainer
    elif args.mode == 'supervised':
        model = ResNetSimCLR(base_model=args.arch, out_dim=len(train_loader.dataset.classes))
        trainer_class = SupervisedTrainer
    else:
        raise InvalidTrainingMode()

    if args.optimizer_mode == 'simclr':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader),
                                                               eta_min=0, last_epoch=-1)

    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        trainer = trainer_class(model=model, optimizer=optimizer, scheduler=scheduler,
                                train_dataset=train_loader.dataset, args=args)
        trainer.train(train_loader, valid_loader)


if __name__ == "__main__":
    main()
