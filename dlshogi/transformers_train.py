import argparse
import importlib
import inspect
import logging
import math
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from dlshogi import cppshogi, serializers
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
from dlshogi.data_loader import DataLoader as HcpeDataLoader
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.network.policy_value_network import policy_value_network


logger = logging.getLogger(__name__)


def _collator(batch):
    if isinstance(batch, dict):
        return batch
    return default_collate(batch)


def _import_object(class_path):
    module_name, object_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _filtered_kwargs(cls, kwargs):
    parameters = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def _coerce_precision(trainer_config):
    precision = trainer_config.get("precision")
    if precision in (16, "16", "16-mixed", "fp16"):
        return {"fp16": True}
    if precision in ("bf16", "bf16-mixed"):
        return {"bf16": True}
    return {}


def _get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def _num_training_batches(train_dataset, per_device_batch_size):
    global_batch_size = per_device_batch_size * _get_world_size()
    return max(1, math.ceil(len(train_dataset) / global_batch_size))


def _steps_from_train_batches(train_batches, gradient_accumulation_steps):
    return max(1, math.ceil(train_batches / gradient_accumulation_steps))


def _resolve_val_check_interval(
    val_check_interval,
    train_dataset,
    per_device_batch_size,
    gradient_accumulation_steps,
    check_val_every_n_epoch,
):
    if val_check_interval is None:
        return None

    epoch_train_batches = _num_training_batches(train_dataset, per_device_batch_size)
    if type(val_check_interval) is float:
        if not 0.0 <= val_check_interval <= 1.0:
            raise ValueError("trainer.val_check_interval as float must be in [0.0, 1.0].")
        train_batches = max(1, int(epoch_train_batches * val_check_interval))
        return _steps_from_train_batches(train_batches, gradient_accumulation_steps)

    if type(val_check_interval) is int:
        if val_check_interval <= 0:
            raise ValueError("trainer.val_check_interval as int must be positive.")
        if (
            check_val_every_n_epoch is not None
            and val_check_interval > epoch_train_batches
        ):
            raise ValueError(
                "trainer.val_check_interval as int can only be higher than the "
                "number of training batches when trainer.check_val_every_n_epoch is null."
            )
        return _steps_from_train_batches(
            val_check_interval, gradient_accumulation_steps
        )

    raise TypeError("trainer.val_check_interval must be an int, float, or null.")


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _resume_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value in ("1", "true", "yes", "on"):
            return True
        if lower_value in ("0", "false", "no", "off"):
            return False
    return value


class HcpeDataset(Dataset):
    def __init__(self, files):
        logger.info("Loading HcpeDataset")
        self.hcpe = HcpeDataLoader.load_files(files, logger)
        logger.info("position num = %d", len(self.hcpe))

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

        return {
            "features1": features1,
            "features2": features2,
            "move": move,
            "result": result,
            "value": value,
        }

    def __getitem__(self, index):
        batch = self.__getitems__([index])
        return {key: value[0] for key, value in batch.items()}


class Hcpe3Dataset(Dataset):
    def __init__(self, files, use_average, use_evalfix, temperature, patch, cache):
        self.files = files
        self.use_average = use_average
        self.use_evalfix = use_evalfix
        self.temperature = temperature
        self.patch = patch
        self.cache = cache
        self._len = None
        self._actual_len = None
        self._loaded_pid = None
        self._ensure_loaded()

    def _ensure_loaded(self):
        pid = os.getpid()
        if self._loaded_pid == pid:
            return

        logger.info("Loading Hcpe3Dataset")
        self._len, self._actual_len = Hcpe3DataLoader.load_files(
            self.files,
            self.use_average,
            self.use_evalfix,
            self.temperature,
            self.patch,
            self.cache,
            logger,
        )
        if self.use_average:
            logger.info("position num before preprocessing = %d", self._actual_len)
        logger.info("position num = %d", self._len)
        self._loaded_pid = pid

    def __len__(self):
        self._ensure_loaded()
        return self._len

    def __getitems__(self, indexes):
        self._ensure_loaded()
        batch_size = len(indexes)
        indexes = np.array(indexes, dtype=np.uint64)

        features1 = torch.empty(
            (batch_size, FEATURES1_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        features2 = torch.empty(
            (batch_size, FEATURES2_NUM, 9, 9), dtype=torch.float32, pin_memory=True
        )
        probability = torch.empty(
            (batch_size, 9 * 9 * MAX_MOVE_LABEL_NUM), dtype=torch.float32, pin_memory=True
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

        return {
            "features1": features1,
            "features2": features2,
            "probability": probability,
            "result": result,
            "value": value,
        }

    def __getitem__(self, index):
        batch = self.__getitems__([index])
        return {key: value[0] for key, value in batch.items()}


def cross_entropy_loss_with_soft_target(pred, soft_targets):
    return torch.sum(-soft_targets * F.log_softmax(pred, dim=1), 1)


class PolicyValueForTrainer(nn.Module):
    def __init__(
        self,
        network="resnet10_relu",
        val_lambda=0.333,
        val_lambda_decay_epoch=None,
        resume_model=None,
        use_compile=False,
        compile_backend=None,
        compile_mode=None,
        compile_fullgraph=False,
        compile_dynamic=False,
        **_,
    ):
        super().__init__()
        self.model = policy_value_network(network)
        self.val_lambda = val_lambda
        self.initial_val_lambda = val_lambda
        self.val_lambda_decay_epoch = val_lambda_decay_epoch

        if resume_model:
            checkpoint = torch.load(resume_model, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state_dict)

        if use_compile:
            compile_kwargs = {}
            if compile_backend is None and os.name == "nt":
                compile_backend = "aot_eager"
            if compile_backend:
                compile_kwargs["backend"] = compile_backend
            if compile_mode:
                compile_kwargs["mode"] = compile_mode
            if compile_fullgraph:
                compile_kwargs["fullgraph"] = True
            if compile_dynamic:
                compile_kwargs["dynamic"] = True
            object.__setattr__(
                self, "_forward_model", torch.compile(self.model, **compile_kwargs)
            )
        else:
            object.__setattr__(self, "_forward_model", self.model)

    def set_epoch(self, epoch):
        if self.val_lambda_decay_epoch:
            self.val_lambda = max(
                0,
                self.initial_val_lambda * (1 - epoch / self.val_lambda_decay_epoch),
            )

    def forward(
        self,
        features1,
        features2,
        probability=None,
        move=None,
        result=None,
        value=None,
        **_,
    ):
        policy_logits, value_logits = self._forward_model(features1, features2)

        loss = None
        policy_loss = None
        result_loss = None
        value_loss = None
        if probability is not None:
            policy_loss = cross_entropy_loss_with_soft_target(
                policy_logits, probability
            ).mean()
        elif move is not None:
            policy_loss = F.cross_entropy(policy_logits, move, reduction="none").mean()

        if result is not None:
            result_loss = F.binary_cross_entropy_with_logits(value_logits, result)
        if value is not None:
            value_loss = F.binary_cross_entropy_with_logits(value_logits, value)

        if policy_loss is not None and result_loss is not None and value_loss is not None:
            loss = (
                policy_loss
                + (1 - self.val_lambda) * result_loss
                + self.val_lambda * value_loss
            )

        return {
            "loss": loss,
            "policy_logits": policy_logits,
            "value_logits": value_logits,
            "policy_loss": policy_loss,
            "result_loss": result_loss,
            "value_loss": value_loss,
        }


class DlshogiTrainer(Trainer):
    def __init__(self, *args, optimizer_config=None, scheduler_config=None, **kwargs):
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if self.optimizer_config:
            optimizer_cls = _import_object(self.optimizer_config["class_path"])
            init_args = self.optimizer_config.get("init_args", {})
            self.optimizer = optimizer_cls(self.model.parameters(), **init_args)
            return self.optimizer
        return super().create_optimizer()

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        if self.scheduler_config:
            scheduler_cls = _import_object(self.scheduler_config["class_path"])
            init_args = dict(self.scheduler_config.get("init_args", {}))
            self.lr_scheduler = scheduler_cls(optimizer or self.optimizer, **init_args)
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        logging_steps = max(1, self.args.logging_steps)
        unwrapped_model = _unwrap_model(model)
        if model.training and self.state.global_step % logging_steps == 0:
            logs = {"train/val_lambda": unwrapped_model.val_lambda}
            for key in ("policy_loss", "result_loss", "value_loss"):
                metric = outputs.get(key)
                if metric is not None:
                    logs[f"train/{key}"] = metric.detach().float().item()
            self.log(logs)
        return (loss, outputs) if return_outputs else loss


class EpochCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and state.epoch is not None:
            _unwrap_model(model).set_epoch(state.epoch)


class SaveNpzCallback(TrainerCallback):
    def __init__(self, model_filename):
        self.model_filename = model_filename

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self.model_filename or not args.should_save or model is None:
            return
        unwrapped = _unwrap_model(model)
        filename = self.model_filename.format(
            epoch=int(state.epoch or 0), step=state.global_step
        )
        serializers.save_npz(os.path.join(args.output_dir, filename), unwrapped.model)


def build_training_arguments(config, train_dataset):
    trainer_config = config.get("trainer", {})
    transformers_config = config.get("transformers", {})
    data_config = config.get("data", {})

    output_dir = transformers_config.get(
        "output_dir",
        trainer_config.get("default_root_dir", os.path.join("runs", "transformers")),
    )
    logging_dir = transformers_config.get("logging_dir", os.path.join(output_dir, "logs"))

    gradient_accumulation_steps = transformers_config.get("gradient_accumulation_steps", 1)
    per_device_train_batch_size = data_config.get("batch_size", 1024)
    eval_steps = transformers_config.get("eval_steps")
    if eval_steps is None:
        eval_steps = _resolve_val_check_interval(
            trainer_config.get("val_check_interval"),
            train_dataset,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            trainer_config.get("check_val_every_n_epoch", 1),
        )
    save_steps = transformers_config.get("save_steps")

    kwargs = {
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "report_to": transformers_config.get("report_to", ["tensorboard"]),
        "num_train_epochs": trainer_config.get("max_epochs", 1),
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": data_config.get(
            "val_batch_size", data_config.get("batch_size", 1024)
        ),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": trainer_config.get("gradient_clip_val", 0.0),
        "logging_steps": transformers_config.get("logging_steps", 50),
        "save_total_limit": transformers_config.get("save_total_limit"),
        "dataloader_num_workers": data_config.get(
            "num_workers", transformers_config.get("dataloader_num_workers", 0)
        ),
        "dataloader_pin_memory": transformers_config.get("dataloader_pin_memory", True),
        "dataloader_persistent_workers": transformers_config.get(
            "dataloader_persistent_workers",
            data_config.get("num_workers", 0) > 0,
        ),
        "prediction_loss_only": transformers_config.get("prediction_loss_only", True),
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": transformers_config.get(
            "ddp_find_unused_parameters", False
        ),
        **_coerce_precision(trainer_config),
    }

    if eval_steps is None:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = eval_steps

    if save_steps is None:
        kwargs["save_strategy"] = transformers_config.get("save_strategy", "epoch")
    else:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = save_steps

    kwargs.update(transformers_config.get("training_arguments", {}))

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in signature and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    return TrainingArguments(**_filtered_kwargs(TrainingArguments, kwargs))


def build_datasets(config):
    data_config = config.get("data", {})
    train_dataset = Hcpe3Dataset(
        data_config["train_files"],
        data_config.get("use_average", False),
        data_config.get("use_evalfix", False),
        data_config.get("temperature", 1.0),
        data_config.get("patch"),
        data_config.get("cache"),
    )
    eval_dataset = HcpeDataset(data_config["val_files"])
    return train_dataset, eval_dataset


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train dlshogi with Hugging Face Transformers Trainer."
    )
    parser.add_argument("command", nargs="?", default="fit", choices=["fit"])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--resume_from_checkpoint")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)

    if "seed_everything" in config:
        set_seed(config["seed_everything"])

    train_dataset, eval_dataset = build_datasets(config)
    training_args = build_training_arguments(config, train_dataset)
    model_config: Dict[str, Any] = dict(config.get("model", {}))
    model_filename = model_config.pop("model_filename", None)
    model = PolicyValueForTrainer(**model_config)

    callbacks = [EpochCallback()]
    if model_filename:
        callbacks.append(SaveNpzCallback(model_filename))

    trainer = DlshogiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_collator,
        optimizer_config=config.get("optimizer"),
        scheduler_config=config.get("lr_scheduler"),
        callbacks=callbacks,
    )

    resume_from_checkpoint = (
        args.resume_from_checkpoint
        or config.get("transformers", {}).get("resume_from_checkpoint")
    )
    resume_from_checkpoint = _resume_value(resume_from_checkpoint)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
