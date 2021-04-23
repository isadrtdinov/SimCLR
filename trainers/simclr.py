import os
import torch
import torch.nn.functional as F
from trainers.basic import BasicTrainer


class SimCLRTrainer(BasicTrainer):
    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for _ in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def calculate_logits(self, images, labels):
        images = torch.cat(images, dim=0)
        images = images.to(self.args.device)

        features = self.model(images)
        logits, labels = self.info_nce_loss(features)
        return logits, labels

    def process_scheduler(self, epoch_counter):
        if epoch_counter >= 10:
            self.scheduler.step()

    def prepare_examples(self, train_dataset):
        full_examples = [[], []]
        for _ in range(self.args.num_augments):
            examples = [[], []]
            for index in self.estimate_indices:
                example = train_dataset[index][0]
                examples[0] += [example[0]]
                examples[1] += [example[1]]

            full_examples[0] += [torch.stack(examples[0], dim=0)]
            full_examples[1] += [torch.stack(examples[1], dim=0)]

        full_examples[0] = torch.stack(full_examples[0], dim=0)
        full_examples[1] = torch.stack(full_examples[1], dim=0)

        return full_examples, None

    @torch.no_grad()
    def estimate_stats(self, epoch_counter):
        full_probs, full_argmax = [], []
        for i in range(self.args.num_augments):
            aug_probs, aug_argmax = [], []

            for j in range(self.args.estimate_batches):
                images = [None, None]
                images[0] = self.estimate_examples[0][i, j * self.args.batch_size:(j + 1) * self.args.batch_size]
                images[1] = self.estimate_examples[1][i, j * self.args.batch_size:(j + 1) * self.args.batch_size]
                logits, labels = self.calculate_logits(images, None)

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
