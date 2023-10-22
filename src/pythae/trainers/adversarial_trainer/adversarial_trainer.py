import datetime
import itertools
import logging
import os
from copy import deepcopy
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..base_trainer import BaseTrainer
from ..trainer_utils import set_seed
from ..training_callbacks import TrainingCallback
from .adversarial_trainer_config import AdversarialTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class AdversarialTrainer(BaseTrainer):
    """Trainer using distinct optimizers for the autoencoder and the discriminator.

    Args:
        model (BaseAE): The model to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_args (AdversarialTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`AdversarialTrainerConfig` is used. Default: None.

        autoencoder_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer`
            used for training the autoencoder. If None, a :class:`~torch.optim.Adam` optimizer is
            used. Default: None.

        discriminator_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer`
            used for training the discriminator. If None, a :class:`~torch.optim.Adam` optimizer is
            used. Default: None.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[AdversarialTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        BaseTrainer.__init__(
            self,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            callbacks=callbacks,
        )

    def set_autoencoder_optimizer(self):
        autoencoder_optimizer_cls = getattr(
            optim, self.training_config.autoencoder_optimizer_cls
        )

        if self.training_config.autoencoder_optimizer_params is not None:
            if self.distributed:
                autoencoder_optimizer = autoencoder_optimizer_cls(
                    itertools.chain(
                        self.model.module.encoder.parameters(),
                        self.model.module.decoder.parameters(),
                    ),
                    lr=self.training_config.autoencoder_learning_rate,
                    **self.training_config.autoencoder_optimizer_params,
                )
            else:
                autoencoder_optimizer = autoencoder_optimizer_cls(
                    itertools.chain(
                        self.model.encoder.parameters(), self.model.decoder.parameters()
                    ),
                    lr=self.training_config.autoencoder_learning_rate,
                    **self.training_config.autoencoder_optimizer_params,
                )
        else:
            if self.distributed:
                autoencoder_optimizer = autoencoder_optimizer_cls(
                    itertools.chain(
                        self.model.module.encoder.parameters(),
                        self.model.module.decoder.parameters(),
                    ),
                    lr=self.training_config.autoencoder_learning_rate,
                )
            else:
                autoencoder_optimizer = autoencoder_optimizer_cls(
                    itertools.chain(
                        self.model.encoder.parameters(), self.model.decoder.parameters()
                    ),
                    lr=self.training_config.autoencoder_learning_rate,
                )

        self.autoencoder_optimizer = autoencoder_optimizer

    def set_autoencoder_scheduler(self):
        if self.training_config.autoencoder_scheduler_cls is not None:
            autoencoder_scheduler_cls = getattr(
                lr_scheduler, self.training_config.autoencoder_scheduler_cls
            )

            if self.training_config.autoencoder_scheduler_params is not None:
                scheduler = autoencoder_scheduler_cls(
                    self.autoencoder_optimizer,
                    **self.training_config.autoencoder_scheduler_params,
                )
            else:
                scheduler = autoencoder_scheduler_cls(self.autoencoder_optimizer)

        else:
            scheduler = None

        self.autoencoder_scheduler = scheduler

    def set_discriminator_optimizer(self):
        discriminator_cls = getattr(
            optim, self.training_config.discriminator_optimizer_cls
        )

        if self.training_config.discriminator_optimizer_params is not None:
            if self.distributed:
                discriminator_optimizer = discriminator_cls(
                    self.model.module.discriminator.parameters(),
                    lr=self.training_config.discriminator_learning_rate,
                    **self.training_config.discriminator_optimizer_params,
                )
            else:
                discriminator_optimizer = discriminator_cls(
                    self.model.discriminator.parameters(),
                    lr=self.training_config.discriminator_learning_rate,
                    **self.training_config.discriminator_optimizer_params,
                )

        else:
            if self.distributed:
                discriminator_optimizer = discriminator_cls(
                    self.model.module.discriminator.parameters(),
                    lr=self.training_config.discriminator_learning_rate,
                )
            else:
                discriminator_optimizer = discriminator_cls(
                    self.model.discriminator.parameters(),
                    lr=self.training_config.discriminator_learning_rate,
                )

        self.discriminator_optimizer = discriminator_optimizer

    def set_discriminator_scheduler(self) -> torch.optim.lr_scheduler:
        if self.training_config.discriminator_scheduler_cls is not None:
            discriminator_scheduler_cls = getattr(
                lr_scheduler, self.training_config.discriminator_scheduler_cls
            )

            if self.training_config.discriminator_scheduler_params is not None:
                scheduler = discriminator_scheduler_cls(
                    self.discriminator_optimizer,
                    **self.training_config.discriminator_scheduler_params,
                )
            else:
                scheduler = discriminator_scheduler_cls(self.discriminator_optimizer)

        else:
            scheduler = None

        self.discriminator_scheduler = scheduler

    def _optimizers_step(self, model_output):
        autoencoder_loss = model_output.autoencoder_loss
        discriminator_loss = model_output.discriminator_loss

        self.autoencoder_optimizer.zero_grad()
        autoencoder_loss.backward(retain_graph=True)

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()

        self.autoencoder_optimizer.step()
        self.discriminator_optimizer.step()

    def _schedulers_step(self, autoencoder_metrics=None, discriminator_metrics=None):
        if self.autoencoder_scheduler is None:
            pass

        elif isinstance(self.autoencoder_scheduler, lr_scheduler.ReduceLROnPlateau):
            self.autoencoder_scheduler.step(autoencoder_metrics)

        else:
            self.autoencoder_scheduler.step()

        if self.discriminator_scheduler is None:
            pass

        elif isinstance(self.discriminator_scheduler, lr_scheduler.ReduceLROnPlateau):
            self.discriminator_scheduler.step(discriminator_metrics)

        else:
            self.discriminator_scheduler.step()

    def prepare_training(self):
        """Sets up the trainer for training"""

        # set random seed
        set_seed(self.training_config.seed)

        # set autoencoder optimizer and scheduler
        self.set_autoencoder_optimizer()
        self.set_autoencoder_scheduler()

        # set discriminator optimizer and scheduler
        self.set_discriminator_optimizer()
        self.set_discriminator_scheduler()

        # create foder for saving
        self._set_output_dir()

        # set callbacks
        self._setup_callbacks()

    def train(self, log_output_dir: str = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        self.prepare_training()

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model_config
        )

        log_verbose = False

        msg = (
            f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
            " - per_device_train_batch_size: "
            f"{self.training_config.per_device_train_batch_size}\n"
            " - per_device_eval_batch_size: "
            f"{self.training_config.per_device_eval_batch_size}\n"
            f" - checkpoint saving every: {self.training_config.steps_saving}\n"
            f"Autoencoder Optimizer: {self.autoencoder_optimizer}\n"
            f"Autoencoder Scheduler: {self.autoencoder_scheduler}\n"
            f"Discriminator Optimizer: {self.discriminator_optimizer}\n"
            f"Discriminator Scheduler: {self.discriminator_scheduler}\n"
        )

        if self.is_main_process:
            logger.info(msg)

        # set up log file
        if log_output_dir is not None and self.is_main_process:
            log_verbose = True
            file_logger = self._get_file_logger(log_output_dir=log_output_dir)

            file_logger.info(msg)

        if self.is_main_process:
            logger.info("Successfully launched training !\n")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10

        for epoch in range(1, self.training_config.num_epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            metrics = {}

            train_losses = self.train_step(epoch)

            [
                epoch_train_loss,
                epoch_train_autoencoder_loss,
                epoch_train_discriminator_loss,
            ] = train_losses
            metrics["train_epoch_loss"] = epoch_train_loss
            metrics["train_autoencoder_loss"] = epoch_train_autoencoder_loss
            metrics["train_discriminator_loss"] = epoch_train_discriminator_loss

            if self.eval_dataset is not None:
                eval_losses = self.eval_step(epoch)

                [
                    epoch_eval_loss,
                    epoch_eval_autoencoder_loss,
                    epoch_eval_discriminator_loss,
                ] = eval_losses
                metrics["eval_epoch_loss"] = epoch_eval_loss
                metrics["eval_autoencoder_loss"] = epoch_eval_autoencoder_loss
                metrics["eval_discriminator_loss"] = epoch_eval_discriminator_loss

                self._schedulers_step(
                    autoencoder_metrics=epoch_eval_autoencoder_loss,
                    discriminator_metrics=epoch_eval_discriminator_loss,
                )

            else:
                epoch_eval_loss = best_eval_loss

                self._schedulers_step(
                    autoencoder_metrics=epoch_train_autoencoder_loss,
                    discriminator_metrics=epoch_train_discriminator_loss,
                )

            if (
                epoch_eval_loss < best_eval_loss
                and not self.training_config.keep_best_on_train
            ):
                best_eval_loss = epoch_eval_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            if (
                self.training_config.steps_predict is not None
                and epoch % self.training_config.steps_predict == 0
                and self.is_main_process
            ):
                true_data, reconstructions, generations = self.predict(best_model)

                self.callback_handler.on_prediction_step(
                    self.training_config,
                    true_data=true_data,
                    reconstructions=reconstructions,
                    generations=generations,
                    global_step=epoch,
                )

            self.callback_handler.on_epoch_end(training_config=self.training_config)

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                if self.is_main_process:
                    self.save_checkpoint(
                        model=best_model, dir_path=self.training_dir, epoch=epoch
                    )
                    logger.info(f"Saved checkpoint at epoch {epoch}\n")

                    if log_verbose:
                        file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log(
                self.training_config,
                metrics,
                logger=logger,
                global_step=epoch,
                rank=self.rank,
            )

        final_dir = os.path.join(self.training_dir, "final_model")

        if self.is_main_process:
            self.save_model(best_model, dir_path=final_dir)
            logger.info("----------------------------------")
            logger.info("Training ended!")
            logger.info(f"Saved final model in {final_dir}")

        if self.distributed:
            dist.destroy_process_group()

        self.callback_handler.on_train_end(self.training_config)

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """
        self.callback_handler.on_eval_step_begin(
            training_config=self.training_config,
            eval_loader=self.eval_loader,
            epoch=epoch,
            rank=self.rank,
        )

        self.model.eval()

        epoch_autoencoder_loss = 0
        epoch_discriminator_loss = 0
        epoch_loss = 0

        for inputs in self.eval_loader:
            inputs = self._set_inputs_to_device(inputs)

            try:
                with torch.no_grad():
                    model_output = self.model(
                        inputs,
                        epoch=epoch,
                        dataset_size=len(self.eval_loader.dataset),
                        uses_ddp=self.distributed,
                    )

            except RuntimeError:
                model_output = self.model(
                    inputs,
                    epoch=epoch,
                    dataset_size=len(self.eval_loader.dataset),
                    uses_ddp=self.distributed,
                )

            autoencoder_loss = model_output.autoencoder_loss
            discriminator_loss = model_output.discriminator_loss

            loss = autoencoder_loss + discriminator_loss

            epoch_autoencoder_loss += autoencoder_loss.item()
            epoch_discriminator_loss += discriminator_loss.item()
            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.training_config)

        epoch_autoencoder_loss /= len(self.eval_loader)
        epoch_discriminator_loss /= len(self.eval_loader)
        epoch_loss /= len(self.eval_loader)

        return epoch_loss, epoch_autoencoder_loss, epoch_discriminator_loss

    def train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        self.callback_handler.on_train_step_begin(
            training_config=self.training_config,
            train_loader=self.train_loader,
            epoch=epoch,
            rank=self.rank,
        )

        # set model in train model
        self.model.train()

        epoch_autoencoder_loss = 0
        epoch_discriminator_loss = 0
        epoch_loss = 0

        for inputs in self.train_loader:
            inputs = self._set_inputs_to_device(inputs)

            model_output = self.model(
                inputs,
                epoch=epoch,
                dataset_size=len(self.train_loader.dataset),
                uses_ddp=self.distributed,
            )

            self._optimizers_step(model_output)

            autoencoder_loss = model_output.autoencoder_loss
            discriminator_loss = model_output.discriminator_loss

            loss = autoencoder_loss + discriminator_loss

            epoch_autoencoder_loss += autoencoder_loss.item()
            epoch_discriminator_loss += discriminator_loss.item()
            epoch_loss += loss.item()

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        if self.distributed:
            self.model.module.update()
        else:
            self.model.update()

        epoch_autoencoder_loss /= len(self.train_loader)
        epoch_discriminator_loss /= len(self.train_loader)
        epoch_loss /= len(self.train_loader)

        return epoch_loss, epoch_autoencoder_loss, epoch_discriminator_loss

    def save_checkpoint(self, model: BaseAE, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here

        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save optimizers
        torch.save(
            deepcopy(self.autoencoder_optimizer.state_dict()),
            os.path.join(checkpoint_dir, "autoencoder_optimizer.pt"),
        )
        torch.save(
            deepcopy(self.discriminator_optimizer.state_dict()),
            os.path.join(checkpoint_dir, "discriminator_optimizer.pt"),
        )

        # save model
        if self.distributed:
            model.module.save(checkpoint_dir)

        else:
            model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")
