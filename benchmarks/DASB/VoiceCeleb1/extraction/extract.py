#!/usr/bin/env/python3
"""Recipe for extracting a discrete tokens with IEMOCAP.

Authors
 * Pooneh Mousavi 2024
"""

# TODO: check resampling part

import os
import sys
import logging
import pathlib as pl
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

print(base_dir)

logger = logging.getLogger(__name__)


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

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    from voxceleb_prepare import prepare_voxceleb  # noqa

        # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_voxceleb,
            kwargs={
                data_folder=hparams["data_folder"],
                save_folder=hparams["save_folder"],
                verification_pairs_file=veri_file_path,
                splits=["train", "dev", "test"],
                split_ratio=[90, 10],
                seg_dur=hparams["sentence_len"],
                skip_prep=hparams["skip_prep"],
                source=hparams["voxceleb_source"]
                if "voxceleb_source" in hparams
                else None,
            },
        )

    tokens_extractor = hparams["tokens_extractor"]
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
        (save_folder / "librispeech").as_posix(),
    )

    if hparams["save_embedding"]:
        save_folder = pl.Path(hparams["save_folder"])
        logger.info(f"Saving embeddings ...")
        tokens_extractor.save_pretrained_embeddings(
            (save_folder / "embeddings").as_posix(),
            vocab_size=hparams["vocab_size"],
            num_codebooks=hparams["num_codebooks"],
        )
