import os
import csv
import logging
import glob
import random
import shutil
import sys
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

def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.info("Starting...")
    logging.info(sys.argv[1:])

    hparams_files, run_opts, _ = sb.parse_arguments(sys.argv[1:])

    print("run_opts")
    print(run_opts)
    print("hparams_files")
    print(hparams_files)
    # Load hyperparameters file with command-line overrides
    hparams = {}
    for hparams_file in hparams_files[0]:
        print(hparams_file)
        with open(hparams_file) as fin:
            hparams_loaded = load_hyperpyyaml(fin)
            #  merge dictionaries
            hparams.update(hparams_loaded)
            print("hparams")
            print(hparams)




if __name__ == "__main__":
    main()
