import sys
sys.path.append('/home/flowers/work/OpenELM')
from src.openelm.utils.code_eval import pool_exec_processes
from utils_label import preprocessing_P3
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.environments.p3.p3 import P3ProbSolResult
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import pickle
import json
from joblib import Parallel, delayed
from openelm.environments.p3.p3 import P3ProbSol_Chat
from openelm.mutation_model import PromptModel
from tqdm import tqdm
import os

from tenacity import *
import instructor

script_dir = os.path.dirname(__file__) 

# @retry(stop=stop_after_attempt(10),wait=wait_random_exponential(min=1, max=40))
def gen_response(prompt):
    return None




max_workers=50
#set to path_embed = None to use all puzzles from trainset


path_embed = "/home/flowers/work/OpenELM/puzzles_train_1.json"#"/home/flowers/work/OpenELM/label_puzzles/preprocess_p3_emb.json"
path_embed_save="/home/flowers/work/OpenELM/preprocess_p3_emb_dedup_puzzles.json"
with initialize(version_base="1.2"):
    cfg = compose(config_name="elmconfig")
    # print(cfg)
config = OmegaConf.to_object(cfg)

cfg_generation: dict = {
            "temperature": 0.,
            "model": config.model.model_path,
        }
print(config.model.model_path)
# n_skills=config.env.n_skills

# client = get_model(config.model)
#init prompt model
mutation_model = PromptModel(config.model)
#init p
env = P3ProbSol_Chat(config= config.env,
        mutation_model=mutation_model)
# instructor_client = instructor.patch(client)
#config.model



out = preprocessing_P3(path_puzzle=path_embed)

# preprocess puzzles
out=out
for i in out:
    del i["f"], i["g"],i["attempts"]
    config.env.GPT_feedback=True
    i["config"] = config.env


list_program_str=[puzz["program_str"] for puzz in out]

print('begin description_filtering')

add_to_results =env.multiple_description_filtering(list_program_str)

print('begin phenotype computation')
list_phenotype_correct_puzzle = env.to_multiple_phenotype(list_program_str)
# print(list_phenotype_correct_puzzle[0])
assert len(list_phenotype_correct_puzzle)==len(add_to_results)
assert len(list_phenotype_correct_puzzle)==len(out)
# assert len(results1)==len(results2), "results1 and results2 should have the same length"
for idx in range(len(out)):  
    # emb = results1[idx].topics.index_topics
    # if not len(emb)<=5: # we should have at most 5 topics
    #     emb=emb[:5]

    out[idx].update(list_phenotype_correct_puzzle[idx])
    out[idx].update(add_to_results[idx])
    out[idx]["fitness"] = 1.
    # out[idx]["interestingness_f"] = results2[idx].interestingness_score_f
    # out[idx]["interestingness_g"] = results2[idx].interestingness_score_g
    # out[idx]["quality"] = (out[idx]["interestingness_f"]+out[idx]["interestingness_g"])/2
    # out[idx]["is_valid"] = results1[idx].puzzle_check.is_valid
    # out[idx]["is_valid_explanation"] = results1[idx].puzzle_check.explanations

list_p3 = [P3ProbSolResult(**p) for p in out]
correct_pb=0
for probsol in list_p3:

    eval_code = (
        f"{probsol.program_str}\n"
        f"def run_eval():\n"
        f"    return f(g())"
    )

    # Run code to see if g6_2 solves f6_2
    result = pool_exec_processes(
        eval_code,
        func_name="run_eval",
        timeout=env.config.timeout,
        processes=env.config.processes,
        debug=True
    )
    if result[0] is True:
        correct_pb+=1
    else:
        print(eval_code)
        
print("correct pb", correct_pb)
print("total_pb",len(list_p3))

list_program_str=[p.__to_dict__() for p in list_p3]

# path = "/media/data/flowers/OpenELM/preprocess_p3.json"
with open(path_embed_save, "w") as f:
    json.dump(list_program_str, f, indent=4)
# #load it

