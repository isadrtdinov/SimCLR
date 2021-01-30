import os
import torch
import torch.backends.cudnn as cudnn
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
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

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    checkpoints = []
    for root, dirs, files in os.walk(os.path.join('wandb', args.wandb_group)):
        if root.endswith('latest-run'):
            continue
        for file in files:
            if file.endswith('.pt'):
                checkpoints += [file]

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    mean_probs = []
    with torch.cuda.device(args.gpu_index):
        for file in checkpoints:
            state = torch.load(file)
            model.load_state_dict(state['state_dict'])
            simclr = SimCLR(model=model, optimizer=optimizer, args=args)

            set_random_seed(args.seed)
            checkpoint_probs = []
            for images, _ in train_loader:
                images = torch.cat(images, dim=0)
                images = images.to(args.device)

                features = model(images)
                logits, _ = simclr.info_nce_loss(features)
                probs = torch.softmax(logits, dim=1)[:, 0]
                checkpoint_probs += [probs]

            checkpoint_probs = torch.cat(checkpoint_probs, dim=0)
            mean_probs += [checkpoint_probs]

    mean_probs = torch.stack(mean_probs, dim=-1)
    mean_probs = torch.mean(mean_probs, dim=-1)

    if not os.path.isdir('experiments'):
        os.mkdir('experiments')
    torch.save(mean_probs, os.path.join('experiments', args.wandb_group))


if __name__ == '__main__':
    main()