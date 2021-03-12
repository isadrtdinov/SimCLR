from trainers.basic import BasicTrainer


class SupervisedTrainer(BasicTrainer):
    def calculate_logits(self, images, labels):
        images = images.to(self.args.device)
        labels = labels.to(self.args.device)
        logits = self.model(images) / self.args.temperature
        return logits, labels

    def process_scheduler(self, epoch_counter):
        self.scheduler.step()
