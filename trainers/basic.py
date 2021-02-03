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

    def train(self, train_loader):
        n_iter = 0
        if not self.args.no_logging:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
            logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, labels in tqdm(train_loader):
                logits, labels = self.calculate_logits(images, labels)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not self.args.no_logging and n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    wandb.log({'loss': loss, 'acc/top1': top1[0], 'acc/top5': top5[0]})

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
