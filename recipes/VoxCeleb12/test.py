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

if __name__ == "__main__":
    main()
