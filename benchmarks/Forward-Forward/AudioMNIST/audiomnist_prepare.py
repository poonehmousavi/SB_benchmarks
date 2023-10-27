"""
Data preparation.
Download: https://voice.mozilla.org/en/datasets
Author
------
Pooneh Mousavi 2023
"""

from dataclasses import dataclass
import os
import logging
import torchaudio
from speechbrain.utils.data_utils import get_all_files
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)
DEFAULT_SPLITS = ["train", "valid", "test"]

def prepare_audiomnist(
    data_folder,
    save_folder,
    train_json=None,
    valid_json=None,
    test_json=None,
    skip_prep=False,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download: https://voice.mozilla.org/en/datasets
    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/<language>/
    save_folder : str
        The directory where to store the csv files.
    train_json : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_json : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_json : str, optional
        Path to the Test Common Voice .tsv file (cs)
    skip_prep: bool
        If True, skip data preparation.
    Example
    -------
    >>> from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 accented_letters, \
                 language="en" \
                 )
    """

    if skip_prep:
        return

    # Check if the target folder exists. Create it if it does not.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    # Create a dictionary with all the data-manifest files.
    json_files = {"train": train_json, "valid": valid_json, "test": test_json}
    if skip(json_files.values()):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # If the dataset doesn't exist yet, download it
    if not check_folders(data_folder):
        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the AudioMNIST dataset)" % (data_folder)
        )
        raise FileNotFoundError(err_msg)

    # List files and create manifest from list
    logger.info(
        f"Creating {train_json}, {valid_json}, and {test_json}"
    )
    extension = [".wav"]
    wav_list = get_all_files(data_folder, match_and=extension)

    # Split the signal list into train, valid, and test sets.
    data_split = create_splits(wav_list)

    for split in DEFAULT_SPLITS:
        create_json(json_files[split],data_split[split])

def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

def skip(json_files):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    json_files : dict
        Dictionary containing the paths where json files will be stored for train, valid, and test.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in json_files:
        if not os.path.isfile(filename):
            return False
    return True




def create_splits(wav_list, split_num=0, task = "digit"):

    """
    Creation of text files specifying which files training, validation and test
    set consist of for each cross-validation split. 

    Parameters:
    -----------
        src: string
            Path to directory containing the directories for each subject that
            hold the preprocessed data in hdf5 format.
        dst: string
            Destination where to store the text files specifying training, 
            validation and test splits.

    """

    logger.info("creating splits")
    splits={"digit":{   "train":[   
                                    # set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2, \
                                          8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),

                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, \
                                         10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),

                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41, \
                                          4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),

                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42, \
                                          5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),

                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1, \
                                          6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54])],

                        "valid":[
                            # set([19,20,21,22]),
                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55])],

                        "test":[  
                            # set([23,24,25,26]),  
                                    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
                                    set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
                                    set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
                                    set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
                                    set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50])]},

            "gender":{  "train":[   set([36, 47, 56, 26, 12, 57, 2, 44, 50, 25, 37, 45]),
                                    set([26, 12, 57, 43, 28, 52, 25, 37, 45, 48, 53, 41]),
                                    set([43, 28, 52, 58, 59, 60, 48, 53, 41, 7, 23, 38]),
                                    set([58, 59, 60, 36, 47, 56, 7, 23, 38, 2, 44, 50])],

                        "valid":[set([43, 28, 52, 48, 53, 41]),
                                    set([58, 59, 60, 7, 23, 38]),
                                    set([36, 47, 56, 2, 44, 50]),
                                    set([26, 12, 57, 25, 37, 45])],

                        "test":[    set([58, 59, 60, 7, 23, 38]),
                                    set([36, 47, 56, 2, 44, 50]),
                                    set([26, 12, 57, 25, 37, 45]),
                                    set([43, 28, 52, 48, 53, 41])]}}
    
    data_split = {}
    for split in DEFAULT_SPLITS:
        audio_split = []
        for audio in wav_list:
            if int(audio.split('/')[-2]) in splits[task][split][split_num]:
                audio_split.append(audio)
        data_split[split] = audio_split 
    return data_split

def create_json(json_file, audiolist):
  json_dict = {}
  logger.info(f"Creating {json_file} ...................")
  for audiofile in tqdm(audiolist):
    # Getting info
    audioinfo = torchaudio.info(audiofile) # Your code here

    # Compute the duration in seconds.
    # This is the number of samples divided by the sampling frequency
    duration = audioinfo.num_frames / audioinfo.sample_rate # Your code here

    # Get digit Label by manipulating the audio path
    digit = audiofile.split('/')[-1].split('_')[0] # Your code here (aim for 1 line)_

    # Get a unique utterance id
    uttid = audiofile.split('/')[-1] # Your code here (aim for 1 line)

    # Create entry for this utterance
    json_dict[uttid] = {
            "wav": audiofile,
            "length": duration,
            "digit": digit,
    }

  # Writing the dictionary to the json file
  with open(json_file, mode="w") as json_f:
    json.dump(json_dict, json_f, indent=2)    
                