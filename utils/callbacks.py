import time
import pytorch_lightning as pl
import wandb


class EfficiencyTimingCallback(pl.Callback):
    def __init__(self):
        self.epoch_start = None
        self.train_start = None
        self.times_per_epoch = []

    def on_fit_start(self, trainer, pl_module):
        self.train_start = time.time()

    def on_fit_end(self, trainer, pl_module):
        total_time = time.time() - self.train_start
        mean_epoch_time = sum(self.times_per_epoch) / len(self.times_per_epoch) if self.times_per_epoch else -1
        std_epoch_time = (sum((x - mean_epoch_time) ** 2 for x in self.times_per_epoch) / len(self.times_per_epoch)) ** 0.5 if self.times_per_epoch else -1
        if wandb.run:
            wandb.log({"train/total_time_sec": total_time, "train/mean_epoch_time_sec": mean_epoch_time, "train/std_epoch_time_sec": std_epoch_time})

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start
        self.times_per_epoch.append(epoch_time)
        if wandb.run:
            wandb.log({"train/epoch_time_sec": epoch_time})

    def on_test_start(self, trainer, pl_module):
        self.test_start = time.time()

    def on_test_end(self, trainer, pl_module):
        total_test_time = time.time() - self.test_start
        if wandb.run:
            wandb.log({"test/total_time_sec": total_test_time})

    def on_validation_start(self, trainer, pl_module):
        self.validation_start = time.time()

    def on_validation_end(self, trainer, pl_module):
        total_validation_time = time.time() - self.validation_start
        if wandb.run:
            wandb.log({"validation/total_time_sec": total_validation_time})
