#!/usr/bin/python3
"""Recipe for training then testing speaker embeddings using the VoxCeleb1 Dataset.

Authors
 * Pooneh Mousavi 2024
"""

import os
from tqdm import tqdm
import sys
import logging
import random
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.distributed import run_on_main

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)


logger = logging.getLogger(__name__)


def compute_embedding(in_toks, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        in_toks, wav_lens = (
            in_toks.to(speaker_brain.device),
            wav_lens.to(speaker_brain.device),
        )
        in_embs = speaker_brain.modules.discrete_embedding_layer(
            in_toks
        )  # [B, T, N-Q, D]

        # Attention-Pooling
        att_w = speaker_brain.modules.attention_mlp(in_embs)  # [B, T, N-Q, 1]
        in_embs = torch.matmul(att_w.transpose(2, -1), in_embs).squeeze(
            -2
        )  # [B, T, D]

        embeddings = speaker_brain.modules.encoder(in_embs, wav_lens)

    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(hparams["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig
            in_toks, _ = batch.speech_tokens

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(hparams["device"]), lens.to(hparams["device"])
            emb = compute_embedding(in_toks, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def get_verification_scores(veri_test):
    """ Computes positive and negative scores given the verification split.
    """
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(hparams["output_folder"], "scores.txt")
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    if "score_norm" in hparams:
        train_cohort = torch.stack(list(train_dict.values()))

    for i, line in enumerate(veri_test):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[0].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        if "score_norm" in hparams:
            # Getting norm stats for enrol impostors
            enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            score_e_c = similarity(enrol_rep, train_cohort)

            if "cohort_size" in hparams:
                score_e_c = torch.topk(
                    score_e_c, k=hparams["cohort_size"], dim=0
                )[0]

            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)

            # Getting norm stats for test impostors
            test_rep = test.repeat(train_cohort.shape[0], 1, 1)
            score_t_c = similarity(test_rep, train_cohort)

            if "cohort_size" in hparams:
                score_t_c = torch.topk(
                    score_t_c, k=hparams["cohort_size"], dim=0
                )[0]

            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # Perform score normalization
        if "score_norm" in hparams:
            if hparams["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif hparams["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif hparams["score_norm"] == "s-norm":
                score_e = (score - mean_e_c) / std_e_c
                score_t = (score - mean_t_c) / std_t_c
                score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores


def dataio_prep_verif(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # 1. Declarations:

    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"], replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=params["n_train_snts"]
    )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    # 1. Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("speech_tokens")
    def tokens_pipeline(id):
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        return tokens

    sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
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

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "speech_tokens"])

    # 4 Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, enrol_dataloader, test_dataloader


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        """
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

        enc_out = self.modules.encoder(in_embs, wav_lens)
        outputs = self.modules.classifier(enc_out)
        return outputs, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.scheduler, "on_batch_end"
        ):
            self.hparams.scheduler.on_batch_end(self.model_optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            if type(self.hparams.scheduler).__name__ in [
                "NewBobScheduler",
                "CyclicLRScheduler",
            ]:
                lr, new_lr = self.hparams.scheduler(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            elif type(self.hparams.scheduler).__name__ == "LinearNoamScheduler":
                lr = self.hparams.scheduler.current_lr
            else:
                raise NotImplementedError

            optimizer = self.optimizer.__class__.__name__
            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "optimizer": optimizer,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"], replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["dev_data"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(
        hparams["original_sample_rate"] * hparams["sentence_len"]
    )

    # 1. Define tokens pipeline:
    tokens_loader = hparams["tokens_loader"]
    num_codebooks = hparams["num_codebooks"]

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("speech_tokens")
    def tokens_pipeline(id):
        tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
        return tokens

    sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["original_sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
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

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "spk_id_encoded", "speech_tokens"]
    )

    return train_data, valid_data, label_encoder


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["cached_data_folder"], os.path.basename(hparams["verification_file"])
    )
    if not os.path.exists(veri_file_path)::
        download_file(hparams["verification_file"], veri_file_path)
    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_voxceleb,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["cached_data_folder"],
                "verification_pairs_file": veri_file_path,
                "splits": ["train", "dev", "test"],
                "split_ratio": [90, 10],
                "seg_dur": hparams["sentence_len"],
                "skip_prep": hparams["skip_prep"],
                "source": hparams["voxceleb_source"]
                if "voxceleb_source" in hparams
                else None,
            },
        )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

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

    # Log number of parameters/buffers
    model_params = sum(
        [
            x.numel()
            for module in hparams["modules"].values()
            for x in module.state_dict().values()
        ]
    )
    hparams["train_logger"].log_stats(
        stats_meta={
            "Model parameters/buffers (M)": f"{model_params / 1e6:.2f}",
        },
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["model_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["enrol_dataloader_opts"],
    )

    if hparams["testing"]:

        # Now preparing for test :
        hparams["device"] = speaker_brain.device

        speaker_brain.modules.eval()
        train_dataloader, enrol_dataloader, test_dataloader = dataio_prep_verif(
            hparams
        )
        # Computing  enrollment and test embeddings
        logger.info("Computing enroll/test embeddings...")

        # First run
        enrol_dict = compute_embedding_loop(enrol_dataloader)
        test_dict = compute_embedding_loop(test_dataloader)

        if "score_norm" in hparams:
            train_dict = compute_embedding_loop(train_dataloader)

        # Compute the EER
        logger.info("Computing EER..")
        # Reading standard verification split
        with open(veri_file_path) as f:
            veri_test = [line.rstrip() for line in f]

        positive_scores, negative_scores = get_verification_scores(veri_test)
        del enrol_dict, test_dict

        eer, th = EER(
            torch.tensor(positive_scores), torch.tensor(negative_scores)
        )
        logger.info("EER(%%)=%f", eer * 100)

        min_dcf, th = minDCF(
            torch.tensor(positive_scores), torch.tensor(negative_scores)
        )
        # Testing
        logger.info("minDCF=%f", min_dcf * 100)
