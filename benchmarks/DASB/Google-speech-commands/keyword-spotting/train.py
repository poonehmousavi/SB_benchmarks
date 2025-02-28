#!/usr/bin/python3
"""Recipe for training a classifier using the
Google Speech Commands v0.02 Dataset.

To run this recipe, use the following command:
> python train.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/xvect.yaml (xvector system)

Author
    * Pooneh Mousavi 2024
"""
import os
import sys
import time
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import logging
import speechbrain as sb
import speechbrain.nnet.CNN
from speechbrain.utils.distributed import run_on_main

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

logger = logging.getLogger(__name__)


class SpeakerBrain(sb.core.Brain):
    """Class for GSC training" """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + command classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
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
            )  # [B, T, D]

        # forward modules
        enc_out = self.modules.encoder(in_embs, wav_lens)
        outputs = self.modules.classifier(enc_out)
        # Ecapa model uses softmax outside of its classifier
        if "softmax" in self.modules.keys():
            outputs = self.modules.softmax(outputs)

        return outputs, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using command-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        command, _ = batch.command_encoded

        # compute the cost function
        loss = self.hparams.compute_cost(predictions, command, lens)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, command, lens)

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

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            if type(self.hparams.scheduler).__name__ == "NewBobScheduler":
                lr, new_lr = self.hparams.scheduler(stats["error_rate"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            elif type(self.hparams.scheduler).__name__ == "LinearNoamScheduler":
                lr = self.hparams.scheduler.current_lr
            else:
                raise NotImplementedError

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    #  Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("speech_tokens")
    def tokens_pipeline(id):
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        return tokens

    sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        info = torchaudio.info(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        resampled = resampled.transpose(0, 1).squeeze(1)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Define text pipeline:
    @sb.utils.data_pipeline.takes("command")
    @sb.utils.data_pipeline.provides("command", "command_encoded")
    def label_pipeline(command):
        yield command
        command_encoded = label_encoder.encode_sequence_torch([command])
        yield command_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    #  Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="command",
    )

    # Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "speech_tokens", "command_encoded"]
    )

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing GSC and annotation into csv files)
    from GSC_prepare import prepare_GSC

    # Known words for V2 12 and V2 35 sets
    if hparams["number_of_commands"] == 12:
        words_wanted = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ]
    elif hparams["number_of_commands"] == 35:
        words_wanted = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "bed",
            "bird",
            "cat",
            "dog",
            "happy",
            "house",
            "marvin",
            "sheila",
            "tree",
            "wow",
            "backward",
            "forward",
            "follow",
            "learn",
            "visual",
        ]
    else:
        raise ValueError("number_of_commands must be 12 or 35")

    # Data preparation
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_GSC,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["cached_data_folder"],
                "validation_percentage": hparams["validation_percentage"],
                "testing_percentage": hparams["testing_percentage"],
                "percentage_unknown": hparams["percentage_unknown"],
                "percentage_silence": hparams["percentage_silence"],
                "words_wanted": words_wanted,
                "skip_prep": hparams["skip_prep"],
            },
        )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

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

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["model_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # with torch.autograd.detect_anomaly():
    # Training
    start_time = time.time()  # Start the timer
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    end_time = time.time()  # End the timer
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    logger.info(f"Model execution time: {elapsed_time:.6f} seconds")

    if hparams["testing"]:
        # Load the best checkpoint for evaluation
        test_stats = speaker_brain.evaluate(
            test_set=test_data,
            min_key="error_rate",
            test_loader_kwargs=hparams["dataloader_options"],
        )
