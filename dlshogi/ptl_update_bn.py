import os
from lightning.pytorch.cli import LightningCLI
from dlshogi.ptl import Model as BaseModel, DataModule, serializers


class Model(BaseModel):
    def on_train_start(self):
        self.tmp_model = self.model
        self.model = self.ema_model
        self.tmp_epoch = self.current_epoch
        self.tmp_step = self.global_step
        return super().on_train_start()

    def on_train_end(self):
        super().on_train_end()
        self.model = self.tmp_model
        del self.tmp_model

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer_closure._backward_fn = None
        optimizer_closure._zero_grad_fn = None
        optimizer_closure()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        pass

    def on_fit_end(self):
        model_filename = self.hparams.model_filename.format(
            epoch=self.tmp_epoch, step=self.tmp_step
        )
        serializers.save_npz(
            os.path.join(self.trainer.log_dir, model_filename),
            self.ema_model,
        )


def main():
    LightningCLI(Model, DataModule)


if __name__ == "__main__":
    main()
