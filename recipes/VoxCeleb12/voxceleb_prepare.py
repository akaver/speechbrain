"""
Read README.md first for download and conversion instructions

Data preparation.
"""

import os
import csv
import logging
import glob
import random
import shutil
import sys  # noqa F401
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchaudio
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


logger = logging.getLogger(__name__)
OPT_FILE = "opt_voxceleb_prepare.pkl"
SAMPLERATE = 16000


def prepare_voxceleb(
        data_folder_voxceleb1,
        data_folder_voxceleb2,
        save_folder,
        test_pairs_in_file=None,
        test_pairs_out_file=None,
        splits=["train", "dev", "test"],
        split_ratio=90,
        seg_dur=3.0,
        amp_th=5e-04,
        split_speaker=False,
        random_segment=False,
        skip_prep=False,
        train_csv ="train.csv",
        valid_csv ="valid.csv",
        enrol_csv ="enrol.csv",
        test_csv ="test.csv",
        file_info="file_info.pkl",
        speaker_quantity=None,
        test_pairs_quantity=None
):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb2.

    Arguments
    ---------
    data_folder_voxceleb1 : str
        Path to the folder where the original VoxCeleb1 dataset is stored.
    data_folder_voxceleb2 : str
        Path to the folder where the original VoxCeleb2 dataset is stored (audio files converted to wav).
    save_folder : str
        The directory where to store the csv files.
    test_pairs_file : str
        txt file containing the test files specification
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    split_speaker : bool
        Speaker-wise split
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.
"""

    if skip_prep:
        logger.info("Skiping dataset preparation by request!")
        return

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder_voxceleb1": data_folder_voxceleb1,
        "data_folder_voxceleb2": data_folder_voxceleb2,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
        "split_speaker": split_speaker,
    }

    if not os.path.exists(save_folder):
        logger.info("Save folder not found, creating dir '{}'".format(save_folder))
        os.makedirs(save_folder)
        if not os.path.exists(save_folder):
            logger.error("Save folder creation failed, returning")
            return

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_file_info = os.path.join(save_folder, file_info)

    logger.info("Creating csv file for the VoxCeleb Dataset..")

    # Split data into train and validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utterance_split_lists(
        data_folder_voxceleb1, data_folder_voxceleb2, save_file_info, split_ratio, test_pairs_in_file,
        speaker_quantity=speaker_quantity
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, train_csv, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, valid_csv, random_segment, amp_th)

    if "test" in splits:
        prepare_csv_enrol_test(
            data_folder_voxceleb1, data_folder_voxceleb2, save_folder, test_pairs_in_file, test_pairs_out_file,
            enrol_csv, test_csv,
            test_pairs_quantity=test_pairs_quantity
        )

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)

    return


# Used for verification split
def _get_utterance_split_lists(
        data_folder_voxceleb1, data_folder_voxceleb2, save_file_info, split_ratio, test_pairs_file, speaker_quantity=None
):
    """
    Tot. number of speakers vox1= 1211.
    Tot. number of speakers vox2= 5994.
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the training and dev set (if any).
    """
    train_lst = []
    dev_lst = []

    data_folders = [
        data_folder_voxceleb1 + '/dev/', data_folder_voxceleb1 + '/test/',
        data_folder_voxceleb2 + '/dev/', data_folder_voxceleb2 + '/test/'
    ]

    test_speakers = []
    if test_pairs_file is not None:
        logger.info("Loading test file list...")
        test_lst = []
        for line in open(test_pairs_file):
            items = line.rstrip("\n").split(" ")
            test_lst.append(items[1])
            test_lst.append(items[2])
        test_lst = set(sorted(test_lst))
        test_speakers = [snt.split("/")[0] for snt in test_lst]
        test_speakers = set(sorted(test_speakers))
        logger.info(f"Unique test speakers (will be excluded from train/dev data): {len(test_speakers)}")

    file_list = []
    # do we already have the info in pickle?
    if not os.path.exists(save_file_info):
        # get all the files from all the locations
        for data_folder in data_folders:
            path = os.path.join(data_folder, "wav", "**", "*.wav")
            files_in_data_folder = glob.glob(path, recursive=True)
            logger.info(f"{path} contains {len(files_in_data_folder)} wav files")
            file_list.extend(files_in_data_folder)
        # save the file list into pickle
        logger.info(f"Saving file list to {save_file_info}")
        save_pkl(file_list, save_file_info)
    else:
        logger.info(f"Loading file list from {save_file_info}")
        file_list = load_pkl(save_file_info)

    logger.info(f"Total {len(file_list)} wav audio files")

    audio_files_dict = {}
    for f in file_list:
        spk_id = f.split("/wav/")[1].split("/")[0]
        if spk_id not in test_speakers:
            audio_files_dict.setdefault(spk_id, []).append(f)

    spk_id_list = list(audio_files_dict.keys())
    random.shuffle(spk_id_list)

    if speaker_quantity is not None:
        logger.warning(f"Using only {speaker_quantity} speakers out of {len(spk_id_list)}")
        spk_id_list = random.sample(spk_id_list, speaker_quantity)

    logger.info(f"Unique speakers found (excluding test speakers) {len(spk_id_list)}")

    full_lst = []
    for spk_id in spk_id_list:
        full_lst.extend(audio_files_dict[spk_id])
    logger.info(f"Audio samples {len(full_lst)}")

    test_size_split = 1 - 0.01 * split_ratio
    train_lst, dev_lst = train_test_split(full_lst, test_size=test_size_split, shuffle=True)
    """
    split = int(0.01 * split_ratio[0] * len(spk_id_list))
    for spk_id in spk_id_list[:split]:
        train_lst.extend(audio_files_dict[spk_id])

    for spk_id in spk_id_list[split:]:
        dev_lst.extend(audio_files_dict[spk_id])
    """

    logger.info(f"Train list {len(train_lst)}, dev list {len(dev_lst)}")

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """

    logger.info(f"Creating csv list: {csv_file}")

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            [spk_id, sess_id, utt_id] = wav_file.split("/")[-3:]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)

        audio_duration = signal.shape[0] / SAMPLERATE
        if random_segment:

            start_sample = 0
            stop_sample = signal.shape[0]

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
        else:
            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)

    csv_output = csv_output + entry
    _write_csv_file(csv_file, csv_output)


def prepare_csv_enrol_test(
        data_folder_voxceleb1, data_folder_voxceleb2, save_folder,
        test_pairs_in_file, test_pairs_out_file,
        save_enrol_csv, save_test_csv, test_pairs_quantity=None):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folder : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.

    Returns
    -------
    None
    """

    # msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    # logger.debug(msg)

    data_folders = [
        data_folder_voxceleb1 + '/dev/', data_folder_voxceleb1 + '/test/',
        data_folder_voxceleb2 + '/dev/', data_folder_voxceleb2 + '/test/'
    ]

    with open(test_pairs_in_file) as fin:
        test_pairs_lines = fin.readlines()
        logger.info(f"Initial test pairs {len(test_pairs_lines)}")

    if test_pairs_quantity is not None and test_pairs_quantity < len(test_pairs_lines):
        test_pairs_lines = random.sample(test_pairs_lines, test_pairs_quantity)

    with open(test_pairs_out_file, "w") as output:
        output.writelines(test_pairs_lines)

    logger.info(f"Final test pairs {len(test_pairs_lines)}")

    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "spk_id"]
    ]  # noqa E231

    # extract all the enrol and test ids
    enrol_ids, test_ids = [], []


    # Get unique ids (enrol and test utterances)
    for line in tqdm(test_pairs_lines):
        splits=line.split(" ")
        e_id = splits[1].rstrip().split(".")[0].strip()
        t_id = splits[2].rstrip().split(".")[0].strip()
        enrol_ids.append(e_id)
        test_ids.append(t_id)

    enrol_ids = list(np.unique(np.array(enrol_ids)))
    test_ids = list(np.unique(np.array(test_ids)))

    enrol_csv = []
    test_csv = []
    logger.info("preparing enrol/test csvs")

    for data_folder in data_folders:
        logger.info(f"Using data from {data_folder}")
        # Prepare enrol csv
        for id in tqdm(enrol_ids):
            wav = data_folder + "wav/" + id + ".wav"

            if not os.path.exists(wav):
                continue

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)

            # Returns a tensor with all the dimensions of input of size 1 removed.
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            enrol_csv.append(csv_line)

        # Prepare test csv
        for id in tqdm(test_ids):
            wav = data_folder + "wav/" + id + ".wav"

            if not os.path.exists(wav):
                continue

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            test_csv.append(csv_line)

    csv_output = csv_output_head + enrol_csv
    csv_file = save_enrol_csv
    # Writing the csv lines
    _write_csv_file(csv_file, csv_output)

    csv_output = csv_output_head + test_csv
    csv_file = save_test_csv
    # Writing the csv lines
    _write_csv_file(csv_file, csv_output)


def _write_csv_file(csv_file, csv_output):
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    logger.info(f"File created: {csv_file}")


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.info("Starting...")


    logging.info(f"Command line args: {sys.argv[1:]}")

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])


    # Load hyperparameters file with command-line overrides
    hparams = {}
    with open(hparams_file) as fin:
        hparams_loaded = load_hyperpyyaml(fin, overrides)
        #  merge yaml dictionaries
        hparams.update(hparams_loaded)

    # load additional yaml files on top of initial config
    if 'yaml' in run_opts:
        for yaml_file in run_opts['yaml'][0]:
            logging.info(f"Loading additional yaml file: {yaml_file}")
            with open(yaml_file) as fin:
                hparams_loaded = load_hyperpyyaml(fin, overrides)
                #  merge yaml dictionaries
                hparams.update(hparams_loaded)

    logging.info(f"Params: {hparams}")


    splits = ['train', 'dev', 'test']

    prepare_voxceleb(
        hparams['data_folder_voxceleb1'], hparams['data_folder_voxceleb2'],
        hparams['data_folder'],

        test_pairs_in_file=hparams['test_pairs_in_file'],
        test_pairs_out_file=hparams['test_pairs_out_file'],

        train_csv=hparams['train_data'], valid_csv= hparams['valid_data'],
        enrol_csv= hparams['enrol_data'], test_csv= hparams['test_data'],

        splits=splits, split_ratio=hparams['split_ratio'],
        speaker_quantity=hparams['speaker_quantity'], test_pairs_quantity=hparams['test_pairs_quantity'],
    )


if __name__ == "__main__":
    main()
