from typing import Iterable, Dict
import torch


def batch_to_device(iterable, device):
    if not isinstance(iterable, Dict):
        keys = range(len(iterable))
        if isinstance(iterable, tuple):
            iterable = list(iterable)
    else:
        keys = iterable

    for k in keys:
        if isinstance(iterable[k], torch.Tensor):
            iterable[k] = iterable[k].to(device)

        elif isinstance(iterable[k], (Iterable, Dict)):
            iterable[k] = batch_to_device(iterable[k], device)

    return iterable


class Trainer:
    def __init__(
        self,
        optimizer,
        min_epochs=0,
        max_epochs=float("inf"),
        limit_train_batches=float("inf"),
        limit_val_batches=float("inf"),
        callbacks=tuple(),
        logger=None,
        print_interval=128,
    ):
        self.optimizer = optimizer
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.logger = logger
        self.callbacks = callbacks
        self.print_interval = print_interval

        self.global_step = 0
        self.current_epoch = 0

    def _add_prefix(self, metrics, prefix):
        return {f"{prefix}/{k}": v for k, v in metrics.items()}

    def train_loop(self, model, train_loader):
        num_iterations = min(len(train_loader), self.limit_val_batches)
        for batch_idx, batch in enumerate(train_loader):
            self.global_step += 1
            loss, metrics = model.training_step(batch, batch_idx)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.print_interval == 0:
                print(
                    f"Step: {self.global_step} Epoch: {self.current_epoch} (Training) [{batch_idx} / {num_iterations}] Loss: {loss.item():.4f}"
                )

            if self.logger is not None:
                self.logger.log_metrics(
                    self._add_prefix(metrics, "train"), step=self.global_step
                )

            if batch_idx >= self.limit_train_batches:
                break

    @torch.no_grad()
    def test_loop(self, model, test_loader):
        test_loss = 0
        test_metrics = dict()
        num_iterations = min(len(test_loader), self.limit_val_batches)
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % self.print_interval == 0:
                print(
                    f"Step: {self.global_step} Epoch: {self.current_epoch} (Testing) [{batch_idx} / {num_iterations}]"
                )
            loss, metrics = model.validation_step(batch, batch_idx)
            test_loss += loss
            for k in metrics:
                if k not in test_metrics:
                    test_metrics[k] = 0
                test_metrics[k] += metrics[k]

            if batch_idx >= self.limit_val_batches:
                break

        test_loss /= num_iterations
        print(f"Test loss: {test_loss:.4f}")

        for k in test_metrics:
            test_metrics[k] /= num_iterations


        test_metrics = self._add_prefix(test_metrics, "val")
        if self.logger is not None:
            # Note: WANDB edits test_metrics inplace.
            self.logger.log_metrics(
                test_metrics, step=self.global_step
            )

        for callback in self.callbacks:
            callback.on_validation_epoch_end(self, model, test_metrics)

    def fit(self, model, train_loader, val_loader):
        self.test_loop(model, val_loader)
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            self.train_loop(model, train_loader)
            self.test_loop(model, val_loader)
