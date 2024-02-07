
import sys
sys.path.append('/home/flowers/work/OpenELM')
# from src.openelm.utils.code_eval import pool_exec_processes
# from utils_label import preprocessing_P3
# from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.environments.p3.p3 import P3ProbSolResult
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import json
from openelm.environments.p3.p3 import P3ProbSol_Chat, P3ProbSol_Chat_PP
from openelm.mutation_model import PromptModel
# from tqdm import tqdm
# import os
from key import OPENAI_API_KEY
import os
from tqdm import tqdm
import numpy as np
from openelm.quality_metrics import utils
import pickle

import argparse
parser = argparse.ArgumentParser(description="argument parsing")
parser.add_argument("-p", "--base_path", type=str, help="path to maps",default="/home/flowers/work/OpenELM/logs/elm/24-02-05_15:39/step_130/maps.pkl")
parser.add_argument("-s", "--save_step", type=str, help="save maps each save_step steps",default=10)

args = parser.parse_args()


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "True"


# load maps
save_step=args.save_step # save each 10 steps

snapshot_path=args.base_path


with open(snapshot_path, "rb") as f:
    maps = pickle.load(f)


fitnesses = maps["fitnesses"]
genomes = maps["genomes"]
non_zeros = maps["nonzero"]

# load config from hydra

with initialize(version_base="1.2"):
    cfg = compose(config_name="elmconfig",overrides=["env=P3ProbSolChatEnv_PP_ELM_NLP"]) # P3ProbSolChatEnv_PP_ELM_NLP
    # print(cfg)
config = OmegaConf.to_object(cfg)

cfg_generation: dict = {
            "temperature": 0.,
            "model": config.model.model_path,
        }
print(config.model.model_path)

#init P3 environment


mutation_model = PromptModel(config.model)
env = P3ProbSol_Chat_PP(config= config.env,
        mutation_model=mutation_model)





disable_tqdm=False

for idx in tqdm(range(len(genomes.archive)),disable=disable_tqdm):
    if hasattr(genomes.archive[idx], "fitnessPP"):
        continue
    puzzle, solution = utils.parse_puzzle_from_str(genomes.archive[idx].program_str)

    final_losses = env._get_losses(puzzle, solution)

    differences = final_losses - env.original_losses
    fitness = differences.mean().item()
    # list_solving_fitness[idx] = - fitness
    genomes.archive[idx].fitnessPP = - fitness
    if idx % save_step ==0:
        maps = {
            "fitnesses": fitnesses,
            "genomes": genomes,
            "nonzero": non_zeros,
        }

        with open(snapshot_path, "wb") as f:
            pickle.dump(genomes, f)

