#!/usr/bin/env python3
"Recipe for training a digit classification system."
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from audiomnist_prepare import prepare_audiomnist
from speechbrain.utils.distributed import run_on_main

# Brain class for speech enhancement training
class DigitBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computations that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """
        # Your code here. Aim for 7-8 lines

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        labels, _ = batch.digit_encoded
        labels = labels.squeeze()

        # Unpacking batch
        wavs, lens = batch.sig

        # Compute features
        feats = self.modules.compute_features(wavs)
        # feats = self.modules.mean_var_norm(feats, lens)


        h_pos , h_neg , hyp = None , None , None
        
        # Final classification
        if stage == sb.Stage.TRAIN:
            x_pos = self.overlay_y_on_x(feats, labels)
            rnd = torch.randperm(feats.size(0))
            x_neg = self.overlay_y_on_x(feats, labels[rnd])
            h_pos, h_neg = self.modules.model.train_layers(x_pos.unsqueeze(1), x_neg.unsqueeze(1))

        else:
            goodness_per_label = []
            for label in range(10):
                h = self.overlay_y_on_x(feats, torch.tensor([label]))
                goodness = self.modules.model.predict(h.unsqueeze(1))
                goodness_per_label += [goodness]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            # hyp = goodness_per_label.argmax(1)
            hyp = goodness_per_label.unsqueeze(1).to(labels.device)

        return h_pos, h_neg , hyp


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        # Your code here. Aim for 7-8 lines
        _, lens = batch.sig
        labels, _ = batch.digit_encoded
        h_pos, h_neg , hyp = predictions

        loss = torch.tensor([0])

        if stage == sb.Stage.TRAIN:                   
            g_pos = h_pos.pow(2).mean(1)
            g_neg = h_neg.pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                    -g_pos + self.modules.model.threshold,
                    g_neg - self.modules.model.threshold]))).mean()

        # # Compute the cost function
        # loss = sb.nnet.losses.nll_loss(predictions, labels)

        # # # Append this batch of losses to the loss metric for easy
        # if stage != sb.Stage.TRAIN:
        #     self.loss_metric.append(
        #         batch.id, predictions, labels, lens, reduction="batch"
        #     )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, hyp, labels, lens)

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

        # # Set up statistics trackers for this stage
        # self.loss_metric = sb.utils.metric_stats.MetricStats(
        #     metric=sb.nnet.losses.nll_loss
        # )

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
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:



            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch,},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
    def overlay_y_on_x(self, x, y):
        """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
        """
        x_ = x.clone()
        # x_[:,0,-10:] *=  0.0
        x_[:, 0:10, 0] = x.min()
        # x_[:, 0, -10:] = x.min()
        # x_[range(x.shape[0]),0, -10+y] = x.max()
        x_[range(x.shape[0]), y, 0] = x.max()
        return x_

    def init_optimizers(self):
        pass
    
    def fit_batch(self, batch):
        out = self.compute_forward(batch, stage=sb.Stage.TRAIN)
        loss = self.compute_objectives(out, batch, stage=sb.Stage.TRAIN)
        return loss.detach().cpu()
    def evaluate_batch(self, batch,stage):
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu() 



def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.
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

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'digit0': 0, 'digit1': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig, fs = torchaudio.load(wav)

        # Resampling
        # Your code here. Aim for 1-2 lines
        sig = torchaudio.functional.resample(sig, fs, 16000)
        sig = sig.squeeze()
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("digit")
    @sb.utils.data_pipeline.provides("digit", "digit_encoded")
    def label_pipeline(digit):
        """Defines the pipeline to process the digit labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        yield digit
        digit_encoded = label_encoder.encode_label_torch(digit)
        yield digit_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    hparams["dataloader_options"]["shuffle"] = True
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "digit_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="digit",
    )

    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin,  overrides)

    # Create experiment directory
    run_on_main(
        prepare_audiomnist,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_save_folder"],
            "train_json": hparams["train_annotation"],
            "valid_json": hparams["valid_annotation"],
            "test_json": hparams["test_annotation"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    digit_brain = DigitBrain(
        modules=hparams["modules"],
        # opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    digit_brain.fit(
        epoch_counter=digit_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = digit_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
