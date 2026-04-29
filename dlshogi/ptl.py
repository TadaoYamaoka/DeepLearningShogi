import logging
import os
from collections import defaultdict

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from lightning.pytorch.cli import LightningCLI
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import DataLoader, Dataset

from dlshogi import cppshogi, serializers
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
from dlshogi.data_loader import DataLoader as HcpeDataLoader
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.network.policy_value_network import policy_value_network


class HcpeDataset(Dataset):
    def __init__(self, files):
        logger = logging.getLogger("lightning.pytorch.core")
        logger.info("Loading HcpeDataset")
        self.hcpe = HcpeDataLoader.load_files(files, logger)
        logger.info("position num = {}".format(len(self.hcpe)))

    def __len__(self):
        return len(self.hcpe)

    def __getitems__(self, indexes):
        batch_size = len(indexes)
        hcpevec = self.hcpe[indexes]

        features1 = torch.empty(
            (batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        features2 = torch.empty(
            (batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        move = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
        result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        cppshogi.hcpe_decode_with_value(
            hcpevec,
            features1.numpy(),
            features2.numpy(),
            move.numpy(),
            result.numpy(),
            value.numpy(),
        )

        return features1, features2, move, result, value


class Hcpe3Dataset(Dataset):
    def __init__(self, files, use_average, use_evalfix, temperature, patch, cache):
        self.files = files
        self.use_average = use_average
        self.use_evalfix = use_evalfix
        self.temperature = temperature
        self.patch = patch
        self.cache = cache
        self.load()

    def load(self):
        logger = logging.getLogger("lightning.pytorch.core")
        logger.info("Loading Hcpe3Dataset")
        self.len, actual_len = Hcpe3DataLoader.load_files(
            self.files,
            self.use_average,
            self.use_evalfix,
            self.temperature,
            self.patch,
            self.cache,
            logger,
        )
        if self.use_average:
            logger.info("position num before preprocessing = {}".format(actual_len))
        logger.info("position num = {}".format(self.len))

    def __len__(self):
        return self.len

    def __getitems__(self, indexes):
        batch_size = len(indexes)
        indexes = np.array(indexes, dtype=np.uint64)

        features1 = torch.empty(
            (batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        features2 = torch.empty(
            (batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        probability = torch.empty(
            (batch_size, 9 * 9 * MAX_MOVE_LABEL_NUM),
            dtype=torch.float32,
            pin_memory=True,
        )
        result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        value = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        cppshogi.hcpe3_decode_with_value(
            indexes,
            features1.numpy(),
            features2.numpy(),
            probability.numpy(),
            result.numpy(),
            value.numpy(),
        )

        return features1, features2, probability, result, value


def collate(data):
    return data


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_files,
        val_files,
        batch_size=1024,
        val_batch_size=1024,
        use_average=False,
        use_evalfix=False,
        temperature=1.0,
        patch=None,
        cache=None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = Hcpe3Dataset(
                self.hparams.train_files,
                self.hparams.use_average,
                self.hparams.use_evalfix,
                self.hparams.temperature,
                self.hparams.patch,
                self.hparams.cache,
            )
            self.val_dataset = HcpeDataset(self.hparams.val_files)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.val_dataset = HcpeDataset(self.hparams.val_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size, collate_fn=collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size, collate_fn=collate
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size, collate_fn=collate
        )


def cross_entropy_loss_with_soft_target(pred, soft_targets):
    return torch.sum(-soft_targets * F.log_softmax(pred, dim=1), 1)


cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()


def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum() / len(t)


def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum() / len(t)


class Model(pl.LightningModule):
    def __init__(
        self,
        network="resnet10_relu",
        val_lambda=0.333,
        val_lambda_decay_epoch=None,
        use_ema=False,
        update_bn=True,
        ema_start_epoch=1,
        ema_freq=250,
        ema_decay=0.9,
        lr_scheduler_interval="epoch",
        model_filename=None,
        resume_model=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = policy_value_network(network)
        if resume_model:
            checkpoint = torch.load(resume_model, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
        if use_ema:
            self.ema_model = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
            )
            self.ema_model.requires_grad_(False)
        self.validation_step_outputs = defaultdict(list)
        self.val_lambda = val_lambda

    def on_train_epoch_start(self):
        # update val_lambda
        if self.hparams.val_lambda_decay_epoch:
            self.val_lambda = max(
                0,
                self.hparams.val_lambda * (1 - self.current_epoch / self.hparams.val_lambda_decay_epoch)
            )
            self.log("val_lambda", self.val_lambda)

    def training_step(self, batch, batch_idx):
        features1, features2, probability, result, value = batch
        y1, y2 = self.model(features1, features2)
        loss1 = cross_entropy_loss_with_soft_target(y1, probability).mean()
        loss2 = bce_with_logits_loss(y2, result)
        loss3 = bce_with_logits_loss(y2, value)
        loss = (
            loss1
            + (1 - self.hparams.val_lambda) * loss2
            + self.hparams.val_lambda * loss3
        )
        self.log("train/loss", loss)
        self.log("train/policy_loss", loss1)
        self.log("train/result_loss", loss2)
        self.log("train/value_loss", loss3)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (
            self.hparams.use_ema
            and self.current_epoch >= self.hparams.ema_start_epoch
            and self.global_step % self.hparams.ema_freq == 0
        ):
            self.ema_model.update_parameters(self.model)

    def on_train_epoch_end(self):
        if (
            self.hparams.use_ema
            and self.hparams.update_bn
            and self.current_epoch == self.trainer.max_epochs - 1
            and self.current_epoch >= self.hparams.ema_start_epoch
        ):

            def data_loader():
                for x1, x2, _, _, _ in Tqdm(
                    self.trainer.datamodule.train_dataloader(),
                    desc="update_bn",
                    dynamic_ncols=True,
                    bar_format=TQDMProgressBar.BAR_FORMAT,
                ):
                    yield {"x1": x1.to(self.device), "x2": x2.to(self.device)}

            forward_ = self.ema_model.forward
            self.ema_model.forward = lambda x: forward_(**x)
            with self.trainer.precision_plugin.train_step_context():
                update_bn(data_loader(), self.ema_model)
            del self.ema_model.forward

    def on_fit_end(self):
        if self.hparams.model_filename:
            if self.hparams.use_ema:
                model = self.ema_model
            else:
                model = self.model
            model_filename = self.hparams.model_filename.format(
                epoch=self.current_epoch, step=self.global_step
            )
            serializers.save_npz(
                os.path.join(self.trainer.log_dir, model_filename),
                model,
            )

    def validation_step(self, batch, batch_idx):
        features1, features2, move, result, value = batch
        y1, y2 = self.model(features1, features2)
        loss1 = cross_entropy_loss(y1, move).mean()
        loss2 = bce_with_logits_loss(y2, result)
        loss3 = bce_with_logits_loss(y2, value)
        loss = (
            loss1
            + (1 - self.hparams.val_lambda) * loss2
            + self.hparams.val_lambda * loss3
        )
        self.validation_step_outputs["val/loss"].append(loss)
        self.validation_step_outputs["val/policy_loss"].append(loss1)
        self.validation_step_outputs["val/result_loss"].append(loss2)
        self.validation_step_outputs["val/value_loss"].append(loss3)

        self.validation_step_outputs["val/policy_accuracy"].append(accuracy(y1, move))
        self.validation_step_outputs["val/value_accuracy"].append(
            binary_accuracy(y2, result)
        )

        entropy1 = (-F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
        self.validation_step_outputs["val/policy_entropy"].append(entropy1.mean())

        p2 = y2.sigmoid()
        # entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
        log1p_ey2 = F.softplus(y2)
        entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
        self.validation_step_outputs["val/value_entropy"].append(entropy2.mean())

    def on_validation_epoch_end(self):
        for key, val in self.validation_step_outputs.items():
            self.log(key, torch.stack(val).mean(), sync_dist=True)
            val.clear()

    def on_test_start(self):
        if self.hparams.use_ema:
            self.tmp_model = self.model
            self.model = self.ema_model
        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        for key, val in self.validation_step_outputs.items():
            key = "test" + key[3:]
            self.log(key, torch.stack(val).mean(), sync_dist=True)
            val.clear()

    def on_test_end(self):
        super().on_test_end()
        if self.hparams.use_ema:
            self.model = self.tmp_model
            del self.tmp_model


class CustomLightningCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        if lightning_module.hparams.lr_scheduler_interval == "epoch":
            return LightningCLI.configure_optimizers(
                lightning_module, optimizer, lr_scheduler
            )
        if lr_scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                **(
                    {"monitor": lr_scheduler.monitor}
                    if isinstance(lr_scheduler, ReduceLROnPlateau)
                    else {}
                ),
            },
        }


def main():
    CustomLightningCLI(Model, DataModule)


if __name__ == "__main__":
    main()
