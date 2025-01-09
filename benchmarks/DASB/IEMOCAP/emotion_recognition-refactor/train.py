#!/usr/bin/env python3
""" Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) starting from a discrete tokens.
The probing head is ECAPA-TDNN.

Authors
 * Pooneh Mousavi 2024
"""

import os
import torch
import torchaudio
import sys
import time
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        in_toks, _ = batch.speech_tokens

        in_embs = self.modules.discrete_embedding_layer(
            in_toks
        )  # [B, T, N-Q, D]

        # Attention-Pooling
        att_w = self.modules.attention_mlp(in_embs)  # [B, T, N-Q, 1]
        in_embs = torch.matmul(att_w.transpose(2, -1), in_embs).squeeze(
            -2
        )  # [B, T, D]

        # forward modules
        if type(self.modules.encoder).__name__ == "ECAPA_TDNN":
            enc_out = self.modules.encoder(in_embs, wav_lens)
            
        elif type(self.modules.encoder).__name__ == "StatisticsPooling":
            enc_out = self.hparams.encoder(in_embs, wav_lens)
            enc_out = encoder.view(enc_out.shape[0], -1)

        else:
            raise NotImplementedError

        outputs = self.modules.classifier(enc_out)
        outputs = self.hparams.log_softmax(enc_out)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        emoid, _ = batch.emo_encoded
        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)
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

            optimizer = self.optimizer.__class__.__name__
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

    # def init_optimizers(self):
    #     "Initializes the weights optimizer and model optimizer"

    #     self.model_optimizer = self.hparams.model_opt_class(
    #         self.hparams.model.parameters()
    #     )
    #     self.optimizers_dict = {
    #         "model_optimizer": self.model_optimizer,
    #     }
    #     # Initializing the weights
    #     if self.checkpointer is not None:
    #         self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


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
    
    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        info = torchaudio.info(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        #         resampled = resampled.unsqueeze(0)
        return resampled

    #  Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("speech_tokens")
    def tokens_pipeline(id):
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        return tokens

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded

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
            output_keys=["id", "sig", "speech_tokens", "emo_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )

    return datasets


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
 
    from iemocap_prepare import prepare_data  # noqa E402

    # Data preparation
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_data,
            kwargs={
                "data_original": hparams["data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
                "split_ratio": [80, 10, 10],
                "different_speakers": hparams["different_speakers"],
                "test_spk_id": hparams["test_spk_id"],
                "seed": hparams["seed"],
            },
        )
    # Data preparation, to be run on only one process.
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

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
    emo_id_brain = EmoIdBrain(
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
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    end_time = time.time()  # End the timer

    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )