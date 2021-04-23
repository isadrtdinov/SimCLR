import os
import logging

import torch
import wandb
from tqdm import tqdm
from utils import accuracy, set_random_seed


class BasicTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        if not os.path.isdir('experiments'):
            os.mkdir('experiments')
        experiment_dir = os.path.join('experiments', self.args.experiment_group)

        self.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        self.stats_dir = os.path.join(experiment_dir, 'stats')
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)
            os.mkdir(self.checkpoint_dir)
            os.mkdir(self.stats_dir)

        set_random_seed(self.args.estimate_seed)
        train_dataset = kwargs['train_dataset']
        self.estimate_indices = torch.randint(len(train_dataset),
                                              size=(self.args.batch_size * self.args.estimate_batches, ))

        examples, labels = self.prepare_examples(train_dataset)
        self.estimate_examples = examples
        self.estimate_labels = labels
        set_random_seed(self.args.seed)

        if not self.args.no_logging:
            wandb.init(project='simclr', config=self.args, group=self.args.experiment_group,
                       dir=experiment_dir)
            wandb.watch(self.model)
            logging.basicConfig(filename=os.path.join(wandb.run.dir, 'training.log'), level=logging.DEBUG)

    def calculate_logits(self, images, labels):
        raise NotImplementedError

    def process_scheduler(self, epoch_counter):
        raise NotImplementedError

    def prepare_examples(self, train_dataset):
        raise NotImplementedError

    def estimate_stats(self, epoch_counter):
        raise NotImplementedError

    def save_checkpoint(self, epoch_counter):
        checkpoint_name = 'checkpoint{}_{:04d}.pt'.format(self.args.seed, epoch_counter)
        torch.save({
            'epoch': epoch_counter,
            'arch': self.args.arch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, checkpoint_name))

    def validate(self, valid_loader, epoch_counter):
        valid_loss, valid_top1, valid_top5 = 0, 0, 0

        for images, labels in valid_loader:
            with torch.no_grad():
                logits, labels = self.calculate_logits(images, labels)
                loss = self.criterion(logits, labels)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))

                valid_loss += loss.item()
                valid_top1 += top1.item()
                valid_top5 += top5.item()

        valid_loss /= len(valid_loader)
        valid_top1 /= len(valid_loader)
        valid_top5 /= len(valid_loader)

        wandb.log({'valid loss': valid_loss, 'valid acc/top1': valid_top1,
                   'valid acc/top5': valid_top5, 'epoch': epoch_counter})

    def train(self, train_loader, valid_loader):
        n_iter = 0
        if not self.args.no_logging:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
            logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(1, self.args.epochs + 1):
            self.model.train()

            for images, labels in tqdm(train_loader):
                logits, labels = self.calculate_logits(images, labels)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not self.args.no_logging and n_iter % self.args.log_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    wandb.log({'train loss': loss.item(),
                               'train acc/top1': top1.item(),
                               'train acc/top5': top5.item(),
                               'train step': n_iter})

                n_iter += 1

            self.process_scheduler(epoch_counter)

            if not self.args.no_logging and epoch_counter % self.args.validation_epochs == 0:
                self.model.eval()
                self.validate(valid_loader, epoch_counter)

            if not self.args.no_logging:
                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss.item()}\tTop1 accuracy: {top1.item()}")

            self.estimate_stats(epoch_counter)
            if not self.args.no_logging and epoch_counter in self.args.checkpoint_epochs:
                self.save_checkpoint(epoch_counter)

        if not self.args.no_logging:
            logging.info("Training has finished.")
            logging.info(f"Model checkpoint and metadata has been saved at {wandb.run.dir}.")
