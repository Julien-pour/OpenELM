"""
This module gives an example of how to run the main ELM class.

It uses the hydra library to load the config from the config dataclasses in
configs.py.

This config file demonstrates an example of running ELM with the Sodarace
environment, a 2D physics-based environment in which robots specified by
Python dictionaries are evolved over.

"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import multiprocessing as mp

import os
os.environ['TRANSFORMERS_CACHE'] = "models"

from openelm import ELM
os.environ["TOKENIZERS_PARALLELISM"] = "True"
                        
os.environ["HYDRA_FULL_ERROR"] = "1"

from transformers import logging
logging.set_verbosity_error()  # avoid all FutureWarnings


@hydra.main(
    config_name="rd_gen",
    version_base="1.2",
)
def main(config):
    path_out=HydraConfig.get().runtime.output_dir
    config.output_dir = path_out
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)
    config.qd.unique_id = config.unique_id+"_s"+str(config.qd.seed)+"_p"

    elm = ELM(config)
    print(
        "Best Individual: ",
        elm.run(init_steps=config.qd.init_steps, total_steps=config.qd.total_steps),
    )


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main()
    