#!/usr/bin/env python3

"""
Common Voice data preparation.

Authors
 * Pooeh Mousavi 2023
"""

import argparse
import csv
import logging
import os
import random
import re
from typing import Optional, Sequence


import torchaudio
from tqdm import tqdm
import soundfile as sf
from os.path import dirname, abspath
from speechbrain.utils.data_utils import download_file

__all__ = [
    "prepare_fleurs",
]


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s - %(message)s",
)

LANGUAGES = {
    "en" : "en_us" ,
    "zh-CN" : "cmn_hans_cn",
    "de": "de_de",
    "es": "gl_es",
    "ru" : "ru_ru",
    "fr" : "fr_fr",
    "pt" : "pt_br",
    "ja" : "ja_jp",
    "tr" : "tr_tr",
    "pl" : "pl_pl",
    "lg" : "lg_ug",
    "ckb" : "ckb_iq",
    "ff" : "ff_sn"
}


# Set backend to SoX (needed to read MP3 files)
if torchaudio.get_audio_backend() != "sox_io":
    torchaudio.set_audio_backend("sox_io")


_LOGGER = logging.getLogger(__name__)




_SPLITS = ["train", "dev", "test"]

# Random indices are not generated on the fly but statically read from a predefined
# file to avoid reproducibility issues on different platforms and/or Python versions
_RANDOM_IDXES_URL = (
    "https://www.dropbox.com/s/v07nprnob0fugoy/random_idxes.txt?dl=1"
)

_RANDOM_IDXES_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "random_idxes.txt"
)

download_file(_RANDOM_IDXES_URL, _RANDOM_IDXES_PATH)

with open(_RANDOM_IDXES_PATH, encoding="utf-8") as f:
    _RANDOM_IDXES = [int(line) for line in f]

# Default seed
random.seed(0)

def prepare_fleurs(
    locales: "Sequence[str]" = ("en",),
    data_folder: "str" = "data",
    max_durations: "Optional[Sequence[float]]" = None,
) -> "None":
    """Prepare the data manifest CSV files for Fleurs dataset
    (see https://huggingface.co/datasets/google/xtreme_s).

    Arguments
    ---------
    locales:
        The locales to use (e.g. "en", "it", etc.).
    data_folder:
        The path to the dataset folder.
    max_durations:
        The maximum total durations in seconds to sample from
        each locale for train, dev and test splits, respectively.
        Default to infinity.

    Raises
    ------
    ValueError
        If an invalid argument value is given.
    RuntimeError
        If a data folder is missing.

    Examples
    --------
    >>> prepare_fluers(["en", "it"], "data")

    """
    if not locales:
        raise ValueError(f"`locales` ({locales}) must be non-empty")
    if max_durations is None:
        max_durations = [None, None, None]
    if len(max_durations) != 3:
        raise ValueError(
            f"`len(max_durations)` ({len(max_durations)}) must be equal to 3"
        )

    for locale in locales:
        locale = LANGUAGES[locale]

        _LOGGER.info(
            "----------------------------------------------------------------------",
        )
        _LOGGER.info(f"Locale: {locale}")
        locale_folder = os.path.join(data_folder, locale)
        if not os.path.isdir(locale_folder):
            raise RuntimeError(
                f'"{locale}" data folder not found. '
                f"Download them from https://huggingface.co/datasets/google/xtreme_s"
            )
        compute_clip_durations(locale_folder)

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info(f"Merging TSV files...")
    for split, max_duration in zip(_SPLITS, max_durations):
        tsv_files = [
            os.path.join(data_folder, LANGUAGES[locale] , f"{split}_with_duration.tsv")
            for locale in locales
        ]
        merge_tsv_files(
            tsv_files,
            os.path.join(data_folder, f"{split}_with_duration.tsv"),
            max_duration,
        )

    _LOGGER.info(
        "----------------------------------------------------------------------",
    )
    _LOGGER.info(f"Creating data manifest CSV files...")
    for split in _SPLITS:
        preprocess_tsv_file(
            os.path.join(data_folder, f"{split}_with_duration.tsv"),
            os.path.join(data_folder, f"{split}.csv"),
        )


def compute_clip_durations(locale_folder: "str") -> "None":
    """Compute clip durations for a FLEURS dataset locale.

    Arguments
    ---------
    locale_folder:
        The path to the dataset locale folder.

    Examples
    --------
    >>> compute_clip_durations("data/en")

    """
    _LOGGER.info("Computing clip durations...")
    locale = locale_folder.split("/")[-1]
    for split in _SPLITS:
        input_tsv_file = os.path.join(locale_folder, f"{split}.tsv")
        output_tsv_file = os.path.join(
            locale_folder, f"{split}_with_duration.tsv"
        )
        if os.path.exists(output_tsv_file):
            _LOGGER.info(f"Clip durations for {split}.tsv already computed")
            continue
        with open(input_tsv_file, encoding="utf-8") as fr, open(
            output_tsv_file, "w", encoding="utf-8"
        ) as fw:
            tsv_reader = csv.reader(fr, delimiter="\t", quoting=csv.QUOTE_NONE)
            tsv_writer = csv.writer(fw, delimiter="\t")
            # header = next(tsv_reader)
            # tsv_writer.writerow(header + ["duration"])
            seen = set() # set for fast O(1) amortized lookup
            for row in tsv_reader:
                if row[0] in seen: 
                    continue # skip duplicate
                # Remove "\t" and "\"" to not confuse the TSV writer
                for i in range(len(row)):
                    row[i] = row[i].replace("\t", " ")
                    row[i] = row[i].replace('"', "")

                mp3 = row[1]
                mp3 = os.path.join(locale_folder, "audio", split , mp3)
                

                # NOTE: info returns incorrect num_frames on torchaudio==0.12.x
                seen.add(row[0])

                # NOTE: info returns incorrect num_frames on torchaudio==0.12.x
                info = torchaudio.info(mp3)
                duration = info.num_frames / info.sample_rate

                tsv_writer.writerow(row + [locale,duration])
    _LOGGER.info("Done!")


def merge_tsv_files(
    input_tsv_files: "Sequence[str]",
    output_tsv_file: "str",
    max_duration: "Optional[float]" = None,
    shuffle: "bool" = False,
) -> "None":
    """Merge input TSV files into a single output TSV file.

    Arguments
    ---------
    input_tsv_files:
        The paths to the input TSV files.
    output_tsv_file:
        The path to the output TSV file.
    max_duration:
        The maximum total duration in seconds to
        sample from each TSV file.
        Default to infinity.
    shuffle:
        True to shuffle the data, False otherwise.
        Used only if `max_duration` is less than infinity.

    Raises
    ------
    IndexError
        If `max_duration` is less than infinity and the
        number of rows in any of the TSV files is larger
        than the maximum allowed (2000000).

    Examples
    --------
    >>> merge_tsv_files(
    ...     ["data/en/test_with_duration.tsv", "data/it/test_with_duration.tsv"],
    ...     "data/test_with_duration.tsv",
    ... )

    """
    if max_duration is None:
        max_duration = float("inf")
    _LOGGER.info(f"Writing output TSV file ({output_tsv_file})...")
    os.makedirs(os.path.dirname(output_tsv_file), exist_ok=True)
    with open(output_tsv_file, "w", encoding="utf-8") as fw:
        tsv_writer = csv.writer(
            fw, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        write_header = False
        for input_tsv_file in input_tsv_files:
            _LOGGER.info(f"Reading input TSV file ({input_tsv_file})...")
            with open(input_tsv_file, encoding="utf-8") as fr:
                tsv_reader = csv.reader(fr, delimiter="\t")
                # header = next(tsv_reader)
                # if write_header:
                #     tsv_writer.writerow(header)
                #     write_header = False
                if max_duration == float("inf"):
                    for row in tsv_reader:
                        tsv_writer.writerow(row)
                    continue
                rows = list(tsv_reader)

            # Add rows until `max_duration` is reached
            random_idxes = (
                random.sample(_RANDOM_IDXES, len(_RANDOM_IDXES))
                if shuffle
                else _RANDOM_IDXES
            )
            duration, i, num_added_rows = 0.0, 0, 0
            while duration <= max_duration and num_added_rows < len(rows):
                try:
                    idx = random_idxes[i]
                except IndexError:
                    raise IndexError(
                        f"The number of rows ({len(rows) + 1}) in {input_tsv_file} "
                        f"must be in the integer interval [1, {len(random_idxes) + 1}]"
                    )
                i += 1
                try:
                    row = rows[idx]
                except IndexError:
                    continue
                duration += float(row[-1])
                num_added_rows += 1
                tsv_writer.writerow(row)
            _LOGGER.info(f"Total duration (s): {duration}")
            _LOGGER.info(f"Added {num_added_rows} rows")

    _LOGGER.info(f"Writing output TSV file ({output_tsv_file})...")
    _LOGGER.info("Done!")


# Adapted from:
# https://github.com/speechbrain/speechbrain/blob/v0.5.13/recipes/CommonVoice/common_voice_prepare.py#L160
def preprocess_tsv_file(
    input_tsv_file: "str", output_csv_file: "str",
) -> "None":
    """Apply minimal Common Voice preprocessing (e.g. rename columns, remove unused columns,
    remove commas, special characters and empty sentences etc.) to each row of an input TSV file.

    Arguments
    ---------
    input_tsv_file:
        The path to the input TSV file.
    output_csv_file:
        The path to the output CSV file.

    Examples
    --------
    >>> preprocess_tsv_file("data/test_with_duration.tsv", "data/test.csv")

    """
    # Header: client_id path sentence up_votes down_votes age gender accents locale segment duration
    _LOGGER.info(f"Reading input TSV file ({input_tsv_file})...")
    _LOGGER.info(f"Writing output CSV file ({output_csv_file})...")
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    num_clips, total_duration = 0, 0.0
    split = os.path.splitext(input_tsv_file)[0].split('/')[-1].split("_")[0]
    with open(input_tsv_file, encoding="utf-8") as fr, open(
        output_csv_file, "w", encoding="utf-8"
    ) as fw:
        tsv_reader = csv.reader(fr, delimiter="\t", quoting=csv.QUOTE_NONE)
        csv_writer = csv.writer(fw)
        _ = next(tsv_reader)
        csv_writer.writerow(["ID", "mp3", "wrd", "locale", "duration"])
        for i, row in enumerate(tsv_reader):
            iden, mp3, wrd, locale, duration = row[0] , row[1], row[3], row[-2], row[-1]
            id_ = os.path.splitext(mp3)[0] + "_" + locale + "_" + iden

            mp3 = os.path.join("$data_root", locale,"audio", split, mp3)
            locale = locale.split('_')[0]

            # Unicode normalization (default in Python 3)
            wrd = str(wrd)

            # Remove commas
            wrd = wrd.replace(",", " ")

            # Replace special characters used by SpeechBrain
            wrd = wrd.replace("$", "S")

            # Remove quotes
            wrd = wrd.replace("'", " ")
            wrd = wrd.replace("’", " ")
            wrd = wrd.replace("`", " ")
            wrd = wrd.replace('"', " ")

            # Remove multiple spaces
            wrd = re.sub(" +", " ", wrd)

            # Remove spaces at the beginning and the end of the sentence
            wrd = wrd.lstrip().rstrip()

            # Remove empty sentences
            if len(wrd) < 1:
                _LOGGER.debug(
                    f"Sentence for row {i + 1} is too short, removing...",
                )
                continue

            # Remove long sentences
            if len(wrd) > 200:
                _LOGGER.debug(
                    f"Sentence for row {i + 1} is too long, removing...",
                )
                continue

            num_clips += 1
            total_duration += float(duration)
            csv_writer.writerow([id_, mp3, wrd, locale, duration])

    with open(f"{output_csv_file}.stats", "w", encoding="utf-8") as fw:
        fw.write(f"Number of samples: {num_clips}\n")
        fw.write(f"Total duration in seconds: {total_duration}")

    _LOGGER.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FLEURS dataset")
    parser.add_argument(
        "-l",
        "--locales",
        nargs="+",
        # fmt: off
        default=[
            "en", "zh-CN", "de", "es", "ru", "fr", "pt", "ja", "tr", "pl",
            "rw", "eo", "kab", "lg", "mhr", "ckb", "ab", "kmr", "fy-NL", "ia",
        ],
        # fmt: off
        help='dataset locales to use (e.g. "en", "it", etc.)',
    )
    parser.add_argument(
        "-d",
        "--data_folder",
        default="CL-MASR",
        help="path to the dataset folder",
    )
    parser.add_argument(
        "-m",
        "--max_durations",
        nargs=3,
        type=float,
        help="maximum total durations in seconds to sample from each "
        "locale for train, dev and test splits, respectively."
        "Default to infinity",
    )

    args = parser.parse_args()
    prepare_fleurs(
        args.locales, args.data_folder, args.max_durations,
    )
