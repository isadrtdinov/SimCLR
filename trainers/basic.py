import os
import logging

import torch
import wandb
from tqdm import tqdm
from utils import accuracy


class BasicTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        if not os.path.isdir('experiments'):
            os.mkdir('experiments')
        experiment_dir = os.path.join('experiments', self.args.experiment_group)
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)

        if not self.args.no_logging:
            wandb.init(project='simclr', config=self.args, group=self.args.experiment_group,
                       dir=experiment_dir)
            wandb.watch(self.model)
            logging.basicConfig(filename=os.path.join(wandb.run.dir, 'training.log'), level=logging.DEBUG)

    def calculate_logits(self, images, labels):
        raise NotImplementedError

    def validate(self, valid_loader):
        valid_loss, valid_top1, valid_top5 = 0, 0, 0

        for images, labels in valid_loader:
            with torch.no_grad():
                logits, labels = self.calculate_logits(images, labels)
                loss = self.criterion(logits, labels)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))

                valid_loss += loss.item()
                valid_top1 += top1[0]
                valid_top5 += top5[0]

        valid_loss /= len(valid_loader)
        valid_top1 /= len(valid_loader)
        valid_top5 /= len(valid_loader)

        wandb.log({'valid loss': valid_loss, 'valid acc/top1': valid_top1,
                   'valid acc/top5': valid_top1})

    def train(self, train_loader, valid_loader):
        n_iter = 0
        if not self.args.no_logging:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
            logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(1, self.args.epochs + 1):
            for images, labels in tqdm(train_loader):
                logits, labels = self.calculate_logits(images, labels)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not self.args.no_logging and n_iter % self.args.log_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    wandb.log({'train loss': loss, 'train acc/top1': top1[0], 'train acc/top5': top5[0]})

                if not self.args.no_logging and n_iter % self.args.validation_steps == 0:
                    self.validate(valid_loader)

                n_iter += 1

            if not self.args.no_logging:
                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            if not self.args.no_logging and epoch_counter in self.args.checkpoint_epochs:
                checkpoint_name = 'checkpoint_{:04d}.pt'.format(epoch_counter)
                torch.save({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, os.path.join(wandb.run.dir, checkpoint_name))

        if not self.args.no_logging:
            logging.info("Training has finished.")
            logging.info(f"Model checkpoint and metadata has been saved at {wandb.run.dir}.")
