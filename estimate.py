import os
import torch
import torch.backends.cudnn as cudnn
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.supervised_learning_dataset import SupervisedLearningDataset
from models.resnet_simclr import ResNetSimCLR
from exceptions.exceptions import InvalidTrainingMode
from trainers.simclr import SimCLRTrainer
from trainers.supervised import SupervisedTrainer
from utils.utils import set_random_seed
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

    checkpoints = []
    for root, dirs, files in os.walk(os.path.join('experiments', args.experiment_group, 'wandb')):
        for file in files:
            if file == args.estimate_checkpoint:
                checkpoints += [os.path.join(root, file)]

    set_random_seed(args.seed)
    sample_indices = torch.randint(len(train_dataset), size=(args.batch_size * args.estimate_batches, ))

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    estimated_prob, estimated_argmax = [], []
    with torch.cuda.device(args.gpu_index):
        for file in checkpoints:
            state = torch.load(file)
            model.load_state_dict(state['model'])
            model.eval()
            trainer = trainer_class(model=model, optimizer=None, args=args)

            checkpoint_prob, checkpoint_argmax = []
            for i in range(args.estimate_batches):
                if args.fixed_augments:
                    set_random_seed(args.seed)

                if args.mode == 'simclr':
                    images = [[], []]
                    for index in sample_indices[i: i + args.batch_size]:
                        example = train_dataset[index][0]
                        images[0] += [example[0]]
                        images[1] += [example[1]]

                    images[0] = torch.stack(images[0], dim=0)
                    images[1] = torch.stack(images[1], dim=0)
                    labels = None
                elif args.mode == 'supervised':
                    images, labels = [], []
                    for index in sample_indices[i: i + args.batch_size]:
                        example = train_dataset[index]
                        images += [example[0]]
                        labels += [example[1]]

                    images = torch.stack(images, dim=0)
                    labels = torch.tensor(labels, dtype=torch.long)

                with torch.no_grad():
                    logits, labels = trainer.calculate_logits(images, labels)

                    prob = torch.softmax(logits, dim=1)[torch.arange(labels.shape[0]), labels]
                    checkpoint_prob += [prob.detach().cpu()]

                    argmax = (torch.argmax(logits, dim=1) == labels).to(torch.int)
                    checkpoint_argmax += [argmax.detach().cpu()]

            checkpoint_prob = torch.cat(checkpoint_prob, dim=0)
            estimated_prob += [checkpoint_prob]

            checkpoint_argmax = torch.cat(checkpoint_argmax, dim=0)
            estimated_argmax += [checkpoint_argmax]

    estimated_prob = torch.stack(estimated_prob, dim=0)
    estimated_argmax = torch.stack(estimated_argmax, dim=0)
    torch.save({
        'indices': sample_indices,
        'prob': estimated_prob,
        'argmax': estimated_argmax
    }, os.path.join('experiments', args.experiment_group, args.out_file))


if __name__ == '__main__':
    main()
