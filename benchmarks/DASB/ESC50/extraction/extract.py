#!/usr/bin/env/python3
"""Recipe for extracting a discrete tokens with Google Speech Command.

Authors
 * Pooneh Mousavi 2024
"""

import os
import sys
import logging
import pathlib as pl
import torchaudio
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

print(base_dir)

logger = logging.getLogger(__name__)


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(wav):
    """Load the signal, and pass it and its length to the corruption class.
    This is done on the CPU in the `collate_fn`."""

    # wave_file = data_audio_folder + "/{:}".format(wav)

    sig, read_sr = torchaudio.load(wav)

    # If multi-channels, downmix it to a mono channel
    sig = torch.squeeze(sig)
    if len(sig.shape) > 1:
        sig = torch.mean(sig, dim=0)

    # Convert sample rate to required config_sample_rate
    resampled = torchaudio.transforms.Resample(
        info.sample_rate, hparams["tokenizer"].sample_rate,
        )(sig)

    resampled = resampled.float()
    resampled = resampled / resampled.max()
            
    return resampled

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing GSC and annotation into csv files)
    from  prepare_esc50 import prepare_esc50

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        run_on_main(
        #     prepare_GSC,
        #     kwargs={
        #         "data_folder": hparams["data_folder"],
        #         "save_folder": hparams["cached_data_folder"],
        #         "validation_percentage": hparams["validation_percentage"],
        #         "testing_percentage": hparams["testing_percentage"],
        #         "percentage_unknown": hparams["percentage_unknown"],
        #         "percentage_silence": hparams["percentage_silence"],
        #         "words_wanted": words_wanted,
        #         "skip_prep": hparams["skip_prep"],
        #     },
        # )
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

    tokens_extractor = hparams["tokens_extractor"]
    tokens_extractor.pipeline_override = audio_pipeline
    data_folder = hparams["data_folder"]
    datasets = []
    for split in ["train", "valid", "test"]:
        json_path = hparams[f"{split}_annotation"]
        name = pl.Path(json_path).stem
        dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path, replacements={"data_root": data_folder},
        )
        datasets.append(dataset)

    merged_data = {
        key: value
        for dataset in datasets
        for key, value in dataset.data.items()
    }
    merged_dataset = DynamicItemDataset(merged_data)

    save_folder = pl.Path(hparams["save_folder"])
    logger.info("Extracting dataset tokens ...")
    tokens_extractor.extract_tokens(
        merged_dataset,
        hparams["num_codebooks"],
        (save_folder / "esc50").as_posix(),
    )

    if hparams["save_embedding"]:
        save_folder = pl.Path(hparams["save_folder"])
        logger.info(f"Saving embeddings ...")
        tokens_extractor.save_pretrained_embeddings(
            (save_folder / "embeddings").as_posix(),
            vocab_size=hparams["vocab_size"],
            num_codebooks=hparams["num_codebooks"],
        )
