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
            info.sample_rate, hparams['tokenizer'].sample_rate,
        )(sig)
        # resampled = resampled.transpose(0, 1).squeeze(1)
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

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_GSC,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_folder": hparams["output_folder"],
                "validation_percentage": hparams["validation_percentage"],
                "testing_percentage": hparams["testing_percentage"],
                "percentage_unknown": hparams["percentage_unknown"],
                "percentage_silence": hparams["percentage_silence"],
                "words_wanted": words_wanted,
                "skip_prep": hparams["skip_prep"],
            },
    )

    tokens_extractor = hparams["tokens_extractor"]
    tokens_extractor.pipeline_override=audio_pipeline
    data_folder = hparams["data_folder"]
    datasets = []
    for split in ["train", "valid", "test"]:
        csv_path = hparams[f"{split}_annotation"]
        name = pl.Path(csv_path).stem
        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path, replacements={"data_root": data_folder},
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
        (save_folder / "gsc").as_posix(),
    )

    if hparams["save_embedding"]:
        save_folder = pl.Path(hparams["save_folder"])
        logger.info(f"Saving embeddings ...")
        tokens_extractor.save_pretrained_embeddings(
            (save_folder / "embeddings").as_posix(),
            vocab_size=hparams["vocab_size"],
            num_codebooks=hparams["num_codebooks"],
        )
