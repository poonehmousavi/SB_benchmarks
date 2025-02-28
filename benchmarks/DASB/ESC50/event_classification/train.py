#!/usr/bin/env python3
""" Recipe for training an event sound classification data only using ESC50.
The system classifies 50 events starting from a discrete tokens.
The probing head is ECAPA-TDNN, linear+pooling.

Authors
 * Pooneh Mousavi 2024
"""
import pdb
import os
import torch
import torchaudio
import sys
import time
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import logging
import glob
import joblib
from sklearn.metrics import confusion_matrix
from confusion_matrix_fig import create_cm_fig
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

logger = logging.getLogger(__name__)


class EventBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        in_toks, _ = batch.speech_tokens

        in_embs = self.modules.discrete_embedding_layer(
            in_toks
        )  # [B, T, N-Q, D]

        # Get merged embedding based on strategy set, defualt Att_Pooling
        if  hasattr(self.hparams,'embedding_strg') and  self.hparams.embedding_strg == 'concat':
            B, T, N_Q, D = in_embs.shape
            in_embs = in_embs.view(B,T,N_Q *D)

        else:
            att_w = self.modules.attention_mlp(in_embs)  # [B, T, N-Q, 1]
            in_embs = torch.matmul(att_w.transpose(2, -1), in_embs).squeeze(
                -2
            )  # [B, T

        # forward modules
        if type(self.modules.encoder).__name__ == "ECAPA_TDNN":
            enc_out = self.modules.encoder(in_embs, wav_lens)

        elif type(self.modules.encoder).__name__ == "StatisticsPooling":
            enc_out = self.modules.encoder(in_embs, wav_lens)
            enc_out = enc_out.view(enc_out.shape[0], -1).unsqueeze(1)

        else:
            raise NotImplementedError

        outputs = self.modules.classifier(enc_out)
        return outputs, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        classid, _ = batch.class_string_encoded

        target = F.one_hot(
            classid.squeeze(), num_classes=self.hparams.out_n_neurons
        )

        loss = (
            -(F.log_softmax(predictions.squeeze(1), 1) * target).sum(1).mean()
        )
        
        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.scheduler, "on_batch_end"):
                self.hparams.scheduler.on_batch_end(self.optimizer)

        self.loss_metric.append(
            uttid, predictions, classid, lens, reduction="batch"
        )
        # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = classid.cpu().detach().numpy().squeeze(-1)
            y_pred = predictions.cpu().detach().numpy().argmax(-1).squeeze(-1)

        if stage == sb.Stage.VALID:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.valid_confusion_matrix += confusion_matix
        if stage == sb.Stage.TEST:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.test_confusion_matrix += confusion_matix

        # Compute accuracy using MetricStats
        self.acc_metric.append(
            uttid, predict=predictions, target=classid, lengths=lens
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, classid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Compute accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target, lengths):
            """Computes accuracy."""
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict, target, lengths
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize("average"),
            }
        # Summarize Valid statistics from the stage for record-keeping
        elif stage == sb.Stage.VALID:
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "error": self.error_metrics.summarize("average"),
            }
        # Summarize Test statistics from the stage for record-keeping
        else:
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "error": self.error_metrics.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, min_keys=["error"]
            )

        # We also write statistics about test data to stdout and to the log file
        if stage == sb.Stage.TEST:
            # Per class accuracy from Test confusion matrix
            per_class_acc_arr = np.diag(self.test_confusion_matrix) / np.sum(
                self.test_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(class_id, class_acc)
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )

            self.hparams.train_logger.log_stats(
                {
                    "Epoch loaded": self.hparams.epoch_counter.current,
                    "\n Per Class Accuracy": per_class_acc_arr_str,
                    "\n Confusion Matrix": "\n{:}\n".format(
                        self.test_confusion_matrix
                    ),
                },
                test_stats=test_stats,
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = wav

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        sig = sig.float()
        sig = sig / sig.max()
        #         resampled = resampled.unsqueeze(0)
        
        return sig
    
    #  Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("speech_tokens")
    def tokens_pipeline(id):
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        return tokens

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        """The label pipeline."""
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, tokens_pipeline, label_pipeline],
            output_keys=["id", "sig", "speech_tokens",  "class_string_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


# RECIPE BEGINS!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from prepare_esc50 import prepare_esc50  # noqa E402

    # Data preparation
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_esc50,
            kwargs={
                "data_folder": hparams["data_folder"],
                "audio_data_folder": hparams["audio_data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
                "train_fold_nums": hparams["train_fold_nums"],
                "valid_fold_nums": hparams["valid_fold_nums"],
                "test_fold_nums": hparams["test_fold_nums"],
                "skip_manifest_creation": hparams["skip_prep"],
            },
        )
    # Data preparation, to be run on only one process.
    # Create dataset objects "train", "valid", and "test".
        # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    # Use pretrained embeddings
    if hparams["pretrain_embeddings"]:
        tokens_loader = hparams["tokens_loader"]
        embs = tokens_loader.load_pretrained_embeddings(
            hparams["pretain_embeddings_folder"]
        )
        if isinstance(hparams["num_codebooks"], int):
            embs = embs[
                : hparams["num_codebooks"] * hparams["vocab_size"],
            ]
        # For discrete SSL, num_codebooks is a list used to determine which layers to use.
        # It is not sequential and can be, for example, [0, 1] or [1, 4].
        elif isinstance(hparams["num_codebooks"], list):
            indices = [
                i
                for codebook_idx in hparams["num_codebooks"]
                for i in range(
                    codebook_idx * hparams["vocab_size"],
                    (codebook_idx + 1) * hparams["vocab_size"],
                )
            ]
            indices = torch.tensor(indices, dtype=torch.long)
            embs = embs[indices]
        hparams["discrete_embedding_layer"].init_embedding(embs)

    # Initialize the Brain object to prepare for mask training.
    event_brain = EventBrain(
        modules=hparams["modules"],
        opt_class=hparams["model_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    # Measure time
    start_time = time.time()  # Start the timer

    event_brain.fit(
        epoch_counter=event_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    end_time = time.time()  # End the timer
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    logger.info(f"Model execution time: {elapsed_time:.6f} seconds")
    if hparams["testing"]:
        # Load the best checkpoint for evaluation
        test_stats = event_brain.evaluate(
            test_set=datasets["test"],
            min_key="error",
            progressbar=True,
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
