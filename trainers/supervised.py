import os
import torch
from trainers.basic import BasicTrainer


class SupervisedTrainer(BasicTrainer):
    def calculate_logits(self, images, labels):
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        logits = self.model(images) / self.args.temperature
        return logits, labels

    def process_scheduler(self, epoch_counter):
        self.scheduler.step()

    def prepare_examples(self, train_dataset):
        full_examples, labels = [], []
        for i in range(self.args.num_augments):
            examples = []
            for index in self.estimate_indices:
                example = train_dataset[index]
                examples += [example[0]]

                if i == 0:
                    labels += [example[1]]

            full_examples += [torch.stack(examples, dim=0)]

        full_examples = torch.stack(full_examples, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return full_examples, labels

    @torch.no_grad()
    def estimate_stats(self, epoch_counter):
        full_probs, full_argmax = [], []
        for i in range(self.args.num_augments):
            aug_probs, aug_argmax = [], []

            for j in range(self.args.estimate_batches):
                images = self.estimate_examples[i, j * self.args.batch_size:(j + 1) * self.args.batch_size]
                labels = self.estimate_labels[j * self.args.batch_size:(j + 1) * self.args.batch_size]
                logits, labels = self.calculate_logits(images, labels)

                probs = torch.softmax(logits, dim=1)[torch.arange(labels.shape[0]), labels]
                aug_probs += [probs.detach().cpu()]

                argmax = (torch.argmax(logits, dim=1) == labels).to(torch.int)
                aug_argmax += [argmax.detach().cpu()]

            full_probs += [torch.cat(aug_probs, dim=0)]
            full_argmax += [torch.cat(aug_argmax, dim=0)]

        full_probs = torch.stack(full_probs, dim=0)
        full_argmax = torch.stack(full_argmax, dim=0)

        stats_file = 'stats{}_{:04d}.pt'.format(self.args.seed, epoch_counter)
        torch.save({
            'indices': self.estimate_indices,
            'prob': full_probs,
            'argmax': full_argmax
        }, os.path.join(self.stats_dir, stats_file))
