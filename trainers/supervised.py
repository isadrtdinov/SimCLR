from trainers.basic import BasicTrainer


class SupervisedTrainer(BasicTrainer):
    def calculate_logits(self, images, labels):
        logits = self.model(images) / self.args.temperature
        return logits, labels
