#!/usr/bin/python
"""
This script implements raining neural networks to decode single EEG trials using various paradigms on MOABB datasets.
For a list of supported datasets and paradigms, please refer to the official documentation at http://moabb.neurotechx.com/docs/api.html.

To run training (e.g., architecture: EEGNet; dataset: BNCI2014001) for a specific subject, recording session and training strategy:
    > python train.py hparams/MotorImagery/BNCI2014001/EEGNet.yaml --data_folder=eeg_data --cached_data_folder=eeg_pickled_data --target_subject_idx=0 --target_session_idx=0 --data_iterator_name=leave-one-session-out

Author
------
Davide Borra, 2022
Mirco Ravanelli, 2023
"""

import datetime
import logging
import os
import pickle
import sys
import uuid

import numpy as np
import speechbrain as sb
import torch
import torch.nn.functional as F
import wandb
import yaml
from hyperpyyaml import load_hyperpyyaml
from pytorch_metric_learning import losses
from torch.nn import init
from torchinfo import summary
from utils.dataio_iterators import LeaveOneSessionOut, LeaveOneSubjectOut


class MOABBBrain(sb.Brain):
    def init_model(self, model):
        """Function to initialize neural network modules"""
        for mod in model.modules():
            if hasattr(mod, "weight"):
                if not ("Norm" in mod.__class__.__name__):
                    init.xavier_uniform_(mod.weight, gain=1)
                else:
                    init.constant_(mod.weight, 1)
            if hasattr(mod, "bias"):
                if mod.bias is not None:
                    init.constant_(mod.bias, 0)

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the model output."
        # B x T x C x 1
        inputs = batch[0].to(self.device)

        if self.hparams.simclr:
            # 2B x T x C x 1 -- needed to create two views of the same signal
            inputs = (
                inputs[:, None]
                .expand(-1, 2, -1, -1, -1)
                .clone()
                .flatten(end_dim=1)
            )

        # Perform data augmentation
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "augment")
            or self.hparams.simclr
        ):
            inputs, _ = self.hparams.augment(
                inputs.squeeze(3),
                lengths=torch.ones(inputs.shape[0], device=self.device),
            )
            inputs = inputs.unsqueeze(3)

        # Normalization
        if hasattr(self.hparams, "normalize"):
            inputs = self.hparams.normalize(inputs)

        return self.modules.model(inputs)

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computes the loss."
        targets = batch[1].to(self.device)
        B = targets.shape[0]

        if not self.hparams.simclr:
            # Target augmentation
            N_augments = int(predictions.shape[0] / targets.shape[0])
            targets = torch.cat(N_augments * [targets], dim=0)

            loss = self.hparams.loss(
                predictions,
                targets,
                weight=torch.FloatTensor(self.hparams.class_weights).to(
                    self.device
                ),
            )

            if stage != sb.Stage.TRAIN:
                # From log to linear predictions
                tmp_preds = torch.exp(predictions)
                self.preds.extend(tmp_preds.detach().cpu().numpy())
                self.targets.extend(batch[1].detach().cpu().numpy())
        else:
            indices = torch.arange(B)[:, None].expand(-1, 2).flatten()
            loss = losses.NTXentLoss()(predictions, indices)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "lr_annealing"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return loss

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``"""
        # Initialize wandb
        run_name = f"exp_bs:{self.hparams.batch_size}_lr:{self.hparams.lr}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=self.hparams.project,
            name=run_name,
            mode=self.hparams.mode,
            entity=self.hparams.entity,
        )
        if self.hparams.random_init:
            # Log configuration settings (hyperparameters) to wandb
            relevant_hparams = {
                "learning_rate": self.hparams.lr,
                "batch_size": self.hparams.batch_size,
                "max_duration_in_seconds": self.hparams.max_duration_in_seconds,
                "model_name_or_path": self.hparams.model_name_or_path,
                "random_init": self.hparams.random_init,
                "freeze": self.hparams.freeze,
                "sampling_rate": self.hparams.sampling_rate,
                "do_stable_layer_norm": self.hparams.do_stable_layer_norm,
                "feat_extract_norm": self.hparams.feat_extract_norm,
                "num_feat_extract_layers": self.hparams.num_feat_extract_layers,
                "conv_dim": self.hparams.conv_dim,
                "conv_kernel": self.hparams.conv_kernel,
                "conv_stride": self.hparams.conv_stride,
                "num_hidden_layers": self.hparams.num_hidden_layers,
                "hidden_size": self.hparams.hidden_size,
                "num_attention_heads": self.hparams.num_attention_heads,
                "intermediate_size": self.hparams.intermediate_size,
                "num_conv_pos_embeddings": self.hparams.num_conv_pos_embeddings,
                "num_codevectors_per_group": self.hparams.num_codevectors_per_group,
            }
            wandb.config.update(relevant_hparams)

        if "skip_init" in hparams and not hparams["skip_init"]:
            self.init_model(self.hparams.model)
        self.init_optimizers()
        in_shape = (
            (1,)
            + tuple(np.floor(self.hparams.input_shape[1:-1]).astype(int))
            + (1,)
        )
        model_summary = summary(
            self.hparams.model, input_size=in_shape, device=self.device
        )
        with open(
            os.path.join(self.hparams.exp_dir, "model.txt"), "w"
        ) as text_file:
            text_file.write(str(model_summary))

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.preds = []
            self.targets = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            self.last_eval_stats = {
                "loss": stage_loss,
            }
            if not self.hparams.simclr:
                preds = np.array(self.preds)
                y_pred = np.argmax(preds, axis=-1)
                y_true = self.targets
                for metric_key in self.hparams.metrics.keys():
                    self.last_eval_stats[metric_key] = self.hparams.metrics[
                        metric_key
                    ](y_true=y_true, y_pred=y_pred)
            if stage == sb.Stage.VALID:
                # Log validation stats to wandb
                wandb.log({"epoch": epoch, **self.last_eval_stats})
                # Learning rate scheduler
                if hasattr(self.hparams, "lr_annealing"):
                    old_lr, new_lr = self.hparams.lr_annealing(epoch)
                    sb.nnet.schedulers.update_learning_rate(
                        self.optimizer, new_lr
                    )
                    self.hparams.train_logger.log_stats(
                        stats_meta={"epoch": epoch, "lr": old_lr},
                        train_stats={"loss": self.train_loss},
                        valid_stats=self.last_eval_stats,
                    )
                else:
                    self.hparams.train_logger.log_stats(
                        stats_meta={"epoch": epoch},
                        train_stats={"loss": self.train_loss},
                        valid_stats=self.last_eval_stats,
                    )

                if epoch == 1:
                    self.best_eval_stats = self.last_eval_stats

                # The current model is saved if it is the best or the last
                is_best = self.check_if_best(
                    self.last_eval_stats,
                    self.best_eval_stats,
                    keys=[self.hparams.test_key],
                )
                is_last = (
                    epoch
                    > self.hparams.number_of_epochs - self.hparams.avg_models
                )

                # Check if we have to save the model
                if self.hparams.test_with == "last" and is_last:
                    save_ckpt = True
                elif self.hparams.test_with == "best" and is_best:
                    save_ckpt = True
                else:
                    save_ckpt = False

                # Saving the checkpoint
                if save_ckpt:
                    min_keys, max_keys = [], []
                    if self.hparams.test_key == "loss":
                        min_keys = [self.hparams.test_key]
                    else:
                        max_keys = [self.hparams.test_key]
                    meta = {}
                    for eval_key in self.last_eval_stats.keys():
                        if eval_key != "cm":
                            meta[str(eval_key)] = float(
                                self.last_eval_stats[eval_key]
                            )
                    self.checkpointer.save_and_keep_only(
                        meta=meta,
                        num_to_keep=self.hparams.avg_models,
                        min_keys=min_keys,
                        max_keys=max_keys,
                    )

            elif stage == sb.Stage.TEST:
                # Log final test metrics to wandb
                wandb.log({**self.last_eval_stats})
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch loaded": self.hparams.epoch_counter.current
                    },
                    test_stats=(
                        self.last_eval_stats
                        if not getattr(self, "log_test_as_valid", False)
                        else None
                    ),
                    valid_stats=(
                        self.last_eval_stats
                        if getattr(self, "log_test_as_valid", False)
                        else None
                    ),
                )
                # save the averaged checkpoint at the end of the evaluation stage
                # delete the rest of the intermediate checkpoints
                # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
                if self.hparams.avg_models > 1:
                    min_keys, max_keys = [], []
                    if self.hparams.test_key == "loss":
                        min_keys = [self.hparams.test_key]
                        fake_meta = {self.hparams.test_key: 0.0, "epoch": epoch}
                    else:
                        max_keys = [self.hparams.test_key]
                        fake_meta = {self.hparams.test_key: 1.1, "epoch": epoch}
                    self.checkpointer.save_and_keep_only(
                        meta=fake_meta,
                        min_keys=min_keys,
                        max_keys=max_keys,
                        num_to_keep=1,
                    )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts,
            recoverable_name="model",
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def check_if_best(
        self,
        last_eval_stats,
        best_eval_stats,
        keys,
    ):
        """Checks if the current model is the best according at least to
        one of the monitored metrics."""
        is_best = False
        for key in keys:
            if key == "loss":
                if last_eval_stats[key] < best_eval_stats[key]:
                    is_best = True
                    best_eval_stats[key] = last_eval_stats[key]
                    break
            else:
                if last_eval_stats[key] > best_eval_stats[key]:
                    is_best = True
                    best_eval_stats[key] = last_eval_stats[key]
                    break
        return is_best


def run_experiment(hparams, run_opts, datasets):
    """This function performs a single training (e.g., single cross-validation fold)"""
    idx_examples = np.arange(datasets["train"].dataset.tensors[0].shape[0])
    n_examples_perclass = [
        idx_examples[
            np.where(datasets["train"].dataset.tensors[1] == c)[0]
        ].shape[0]
        for c in range(hparams["n_classes"])
    ]
    n_examples_perclass = np.array(n_examples_perclass)
    class_weights = n_examples_perclass.max() / n_examples_perclass
    hparams["class_weights"] = class_weights

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=os.path.join(hparams["exp_dir"], "save"),
        recoverables={
            "model": hparams["model"],
            "counter": hparams["epoch_counter"],
        },
    )
    hparams["train_logger"] = sb.utils.train_logger.FileTrainLogger(
        save_file=os.path.join(hparams["exp_dir"], "train_log.txt")
    )
    logger = logging.getLogger(__name__)
    logger.info("Experiment directory: {0}".format(hparams["exp_dir"]))
    logger.info(
        "Input shape: {0}".format(
            datasets["train"].dataset.tensors[0].shape[1:]
        )
    )
    logger.info(
        "Training set avg value: {0}".format(
            datasets["train"].dataset.tensors[0].mean()
        )
    )
    datasets_summary = "Number of examples: {0} (training), {1} (validation), {2} (test)".format(
        datasets["train"].dataset.tensors[0].shape[0],
        datasets["valid"].dataset.tensors[0].shape[0],
        datasets["test"].dataset.tensors[0].shape[0],
    )
    logger.info(datasets_summary)

    brain = MOABBBrain(
        modules={"model": hparams["model"]},
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    # training
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        progressbar=False,
    )
    # evaluation after loading model using specific key
    perform_evaluation(brain, hparams, datasets, dataset_key="test")
    # After the first evaluation only 1 checkpoint (best overall or averaged) is stored.
    # Setting avg_models to 1 to prevent deleting the checkpoint in subsequent calls of the evaluation stage.
    brain.hparams.avg_models = 1
    perform_evaluation(brain, hparams, datasets, dataset_key="valid")


def perform_evaluation(brain, hparams, datasets, dataset_key="test"):
    """This function perform the evaluation stage on a dataset and save the performance metrics in a pickle file"""
    brain.log_test_as_valid = dataset_key == "valid"

    min_key, max_key = None, None
    if hparams["test_key"] == "loss":
        min_key = hparams["test_key"]
    else:
        max_key = hparams["test_key"]
    # perform evaluation
    brain.evaluate(
        datasets[dataset_key],
        progressbar=False,
        min_key=min_key,
        max_key=max_key,
    )
    # saving metrics on the desired dataset in a pickle file
    metrics_fpath = os.path.join(
        hparams["exp_dir"], "{0}_metrics.pkl".format(dataset_key)
    )
    with open(metrics_fpath, "wb") as handle:
        pickle.dump(
            brain.last_eval_stats, handle, protocol=pickle.HIGHEST_PROTOCOL
        )


def prepare_dataset_iterators(hparams):
    """Preprocesses the dataset and partitions it into train, valid and test sets."""
    # defining data iterator to use
    print("Prepare dataset iterators...")
    if hparams["data_iterator_name"] == "leave-one-session-out":
        data_iterator = LeaveOneSessionOut(
            seed=hparams["seed"]
        )  # within-subject and cross-session
    elif hparams["data_iterator_name"] == "leave-one-subject-out":
        data_iterator = LeaveOneSubjectOut(
            seed=hparams["seed"]
        )  # cross-subject and cross-session
    else:
        raise ValueError(
            "Unknown data_iterator_name: %s" % hparams["data_iterator_name"]
        )

    tail_path, datasets = data_iterator.prepare(
        data_folder=hparams["data_folder"],
        dataset=hparams["dataset"],
        cached_data_folder=hparams["cached_data_folder"],
        batch_size=hparams["batch_size"],
        valid_ratio=hparams["valid_ratio"],
        target_subject_idx=hparams["target_subject_idx"],
        target_session_idx=hparams["target_session_idx"],
        events_to_load=hparams["events_to_load"],
        original_sample_rate=hparams["original_sample_rate"],
        sample_rate=hparams["sample_rate"],
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
        tmin=hparams["tmin"],
        tmax=hparams["tmax"],
        save_prepared_dataset=hparams["save_prepared_dataset"],
        n_steps_channel_selection=hparams["n_steps_channel_selection"],
    )
    return tail_path, datasets


def find_best_model_ckpt(base_path):
    best_loss = float("inf")
    best_model_ckpt_path = None

    for root, dirs, files in os.walk(base_path):
        if "CKPT.yaml" in files:
            ckpt_path = os.path.join(root, "CKPT.yaml")
            model_ckpt_path = os.path.join(root, "model.ckpt")

            with open(ckpt_path, "r") as f:
                ckpt_data = yaml.safe_load(f)

            loss = ckpt_data.get("loss", float("inf"))
            if loss < best_loss:
                best_loss = loss
                best_model_ckpt_path = model_ckpt_path

    return best_model_ckpt_path


def load_hparams_and_dataset_iterators(hparams_file, run_opts, overrides):
    """Loads the hparams and datasets, injecting appropriate overrides
    for the shape of the dataset.
    """
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    tail_path, datasets = prepare_dataset_iterators(hparams)
    # override C and T, to be sure that network input shape matches the dataset (e.g., after time cropping or channel sampling)
    overrides.update(
        T=datasets["train"].dataset.tensors[0].shape[1],
        C=datasets["train"].dataset.tensors[0].shape[-2],
        n_train_examples=datasets["train"].dataset.tensors[0].shape[0],
    )

    # loading hparams for the each training and evaluation processes
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if not hparams["simclr"] and hparams["ft"]:
        # do not update the model
        print(f"+++++++ Loading pretrained EEGNet +++++++++")
        # hparams["model"].conv_module.requires_grad_(False)
        # load the encoder class_weights
        best_path = find_best_model_ckpt(hparams["simclr_pretrained"])
        print(f"++++ Best checkpoint found in {best_path} ++++++")

        tmp = torch.load(best_path)
        tmp_keys = list(tmp.keys())
        for k in tmp_keys:
            if "simclr" in k:
                del tmp[k]
        print("Loaded keys: ", list(tmp.keys()))
        hparams["model"].load_state_dict(tmp, strict=True)

    if hparams["sweep_run"]:
        # Extract relevant hyperparameters for path creation
        hidden_size = overrides.get("hidden_size", str(uuid.uuid4()))
        num_attention_heads = overrides.get(
            "num_attention_heads", str(uuid.uuid4())
        )
        # Generate a random identifier
        unique_id = str(uuid.uuid4())
        unique_output_folder = os.path.join(
            hparams["output_folder"],
            f"{hidden_size}_{num_attention_heads}_{unique_id}",
            tail_path,
        )
        hparams["exp_dir"] = unique_output_folder
    else:
        hparams["exp_dir"] = os.path.join(hparams["output_folder"], tail_path)

    # creating experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["exp_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    return hparams, datasets


if __name__ == "__main__":
    argv = sys.argv[1:]
    # loading hparams to prepare the dataset and the data iterators
    hparams_file, run_opts, overrides = sb.core.parse_arguments(argv)
    overrides = yaml.load(
        overrides, yaml.SafeLoader
    )  # Convert overrides to a dict
    hparams, datasets = load_hparams_and_dataset_iterators(
        hparams_file, run_opts, overrides
    )

    # Run training
    run_experiment(hparams, run_opts, datasets)
