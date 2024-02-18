
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
import os
from tqdm import tqdm
import numpy as np
from openelm.quality_metrics import utils
import pickle

import argparse
parser = argparse.ArgumentParser(description="argument parsing")
parser.add_argument("-p", "--puzzle_path", type=str, help="path to puzzles", 
                    default=None)
parser.add_argument("-s", "--save_step", type=str, 
                    help="save maps each save_step steps", default=10)
parser.add_argument('-b', '--batch-size', help='override the batch size of the env',
                    default=2, type=int)

args = parser.parse_args()

try:
    os.environ["OPENAI_API_KEY"]
except IndexError:
    from key import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TOKENIZERS_PARALLELISM"] = "True"


# load maps
save_step = args.save_step

# snapshot_path = args.puzzle_path

# with open(snapshot_path, "r") as f:
#     puzzles = json.load(f)

# open puzzles
all_puzzles = []

save_path = 'compute_pp.json'

if args.puzzle_path is not None:
    save_path = args.puzzle_path.replace('.json', '_compute_pp.json')
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            all_puzzles = json.load(f)
    
    else:
        with open(args.puzzle_path, 'r') as f:
            all_puzzles = json.load(f)

else:
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            all_puzzles = json.load(f)

    else:
        paths = [
            'puzzles_train.json',
            'puzzles_test.json',
            'logs/latest/puzzles_ELM_NLP_0.json',
            'logs/latest/puzzles_ACES_smart_0.json',
            'logs/latest/puzzles_rd_gen_0.json',
        ]

        for path in paths:
            with open(path, 'r') as f:
                d = json.load(f)
                for p in d:
                    p['origin'] = path
            all_puzzles += d
    

with initialize(version_base="1.2"):
    cfg = compose(
        config_name="elmconfig",
        overrides=["env=P3ProbSolChatEnv_PP_ELM_NLP"]
    )
    # print(cfg)
config = OmegaConf.to_object(cfg)

cfg_generation: dict = {
            "temperature": 0.,
            "model": config.model.model_path,
        }
print(config.model.model_path)

# init P3 environment

mutation_model = PromptModel(config.model)
env = P3ProbSol_Chat_PP(config=config.env,
        mutation_model=mutation_model)

env.batch_size = args.batch_size


disable_tqdm = False
all_puzzles[0]["original_losses"] = env.original_losses.cpu().tolist()


for idx in tqdm(range(len(all_puzzles)), disable=disable_tqdm):
    if "fitnessPP" in  all_puzzles[idx] and "final_losses" in all_puzzles[idx]:
        continue
    try:
        puzzle, solution = utils.get_puzzle_sol(all_puzzles[idx])
    except IndexError:  # puzzle has no solution
        continue

    if len(puzzle) > 1000:
        continue 

    final_losses = env._get_losses(puzzle, solution)

    differences = final_losses - env.original_losses
    fitness = differences.mean().item()
    # list_solving_fitness[idx] = - fitness
    all_puzzles[idx]["fitnessPP"] = - fitness
    all_puzzles[idx]["final_losses"] = final_losses.cpu().tolist()
    if (idx+1) % save_step ==0:
        # with open(snapshot_path+"/puzzles.json", "w") as f:
        #     json.dump(puzzles, f, indent=4)
        with open(save_path, "w") as f:
            json.dump(all_puzzles, f, indent=4)