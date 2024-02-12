
import sys
sys.path.append('/home/flowers/work/OpenELM')

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
parser.add_argument("-p", "--base_path", type=str, help="path to maps",default="/home/flowers/work/OpenELM/logs/elm/24-02-05_15:39/step_130")
parser.add_argument("-s", "--save_step", type=str, help="save maps each save_step steps",default=2)

args = parser.parse_args()


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "True"


# load maps
save_step=args.save_step # save each 10 steps

snapshot_path=args.base_path

with open(snapshot_path+"/puzzles.json", "r") as f:
    puzzles = json.load(f)
# with open(snapshot_path+"/puzzles_test.json", "r") as f:
#     puzzles = json.load(f)

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
puzzles[0]["original_losses"]= env.original_losses.cpu().tolist()
for idx in tqdm(range(len(puzzles)),disable=disable_tqdm):
    if "fitnessPP" in  puzzles[idx] and "final_losses" in puzzles[idx]:
        continue
    puzzle, solution = utils.parse_puzzle_from_str(puzzles[idx]["program_str"])

    final_losses = env._get_losses(puzzle, solution)

    differences = final_losses - env.original_losses
    fitness = differences.mean().item()
    # list_solving_fitness[idx] = - fitness
    puzzles[idx]["fitnessPP"] = - fitness
    puzzles[idx]["final_losses"] = final_losses.cpu().tolist()
    if (idx+1) % save_step ==0:
        # with open(snapshot_path+"/puzzles.json", "w") as f:
        #     json.dump(puzzles, f, indent=4)
        with open(snapshot_path+"/puzzles_test.json", "w") as f:
            json.dump(puzzles, f, indent=4)