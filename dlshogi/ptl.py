from collections import defaultdict

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset

from dlshogi import cppshogi
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
from dlshogi.data_loader import DataLoader as HcpeDataLoader
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.network.policy_value_network import policy_value_network


class HcpeDataset(Dataset):
    def __init__(self, files):
        self.hcpe = HcpeDataLoader.load_files(files)

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
        self.len, self.actual_len = Hcpe3DataLoader.load_files(
            self.files,
            self.use_average,
            self.use_evalfix,
            self.temperature,
            self.patch,
            self.cache,
        )

    def __len__(self):
        return self.len

    def __getitems__(self, indexes):
        batch_size = len(indexes)
        indexes = np.array(indexes, dtype=np.uint32)

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
            self.test_dataset = HcpeDataset(self.hparams.val_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.val_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.val_batch_size)


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
    def __init__(self, network="resnet10_relu", val_lambda=0.333):
        super().__init__()
        self.save_hyperparameters()
        self.model = policy_value_network(network)
        self.validation_step_outputs = defaultdict(list)

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
        return loss

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
            self.log(key, torch.stack(val).mean())
            val.clear()


def main():
    LightningCLI(Model, DataModule)


if __name__ == "__main__":
    main()
