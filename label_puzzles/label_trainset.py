import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
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
from openelm.environments.p3 import get_programming_puzzles_prompt,Puzzle_Diversity,Puzzle_Interestingness,create_prompt_label,skill_list
from openelm.mutation_model import get_model,get_multiple_completions_instructor
from tqdm import tqdm
import os
from tenacity import *
import instructor

script_dir = os.path.dirname(__file__) 

# @retry(stop=stop_after_attempt(10),wait=wait_random_exponential(min=1, max=40))
def gen_response(prompt):
    return None


    



max_workers=40
path_embed = "/home/flowers/work/OpenELM/src/openelm/utils/preprocess_p3_emb.json"#"/home/flowers/work/OpenELM/label_puzzles/preprocess_p3_emb.json"#script_dir+"/src/openelm/utils/preprocess_p3_emb.json"

with initialize(version_base="1.2"):
    cfg = compose(config_name="elmconfig")
    # print(cfg)
config = OmegaConf.to_object(cfg)

cfg_generation: dict = {
            "temperature": 0.,
            "top_p": config.model.top_p,
            "model": config.model.model_path,
        }
n_skills=config.env.n_skills

client = get_model(config.model)
instructor_client = instructor.patch(client)
#config.model



out = preprocessing_P3()

# preprocess puzzles
out=out
for i in out:
    del i["f"], i["g"],i["attempts"]
    config.env.GPT_feedback=True
    i["config"] = config.env

batch_prompt1=[create_prompt_label(puzz["program_str"]) for puzz in out]
batch_tools1=[Puzzle_Diversity for _ in range(len(batch_prompt1))]
batch_prompt2=[create_prompt_label(puzz["program_str"],Puzzle_Interestingness=True) for puzz in out]
batch_tools2=[Puzzle_Interestingness for _ in range(len(batch_prompt1))]

results1 = get_multiple_completions_instructor(client, batch_prompt = batch_prompt1, cfg_generation=cfg_generation, batch_tools= batch_tools1,max_workers=max_workers,temperature=0.0)
results2 = get_multiple_completions_instructor(client, batch_prompt = batch_prompt2, cfg_generation=cfg_generation, batch_tools= batch_tools2,max_workers=max_workers,temperature=0.0)
assert len(results1)==len(results2), "results1 and results2 should have the same length"
for idx in range(len(results1)):  
    emb = results1[idx].topics.index_topics
    if not len(emb)<=5: # we should have at most 5 topics
        emb=emb[:5]

    emb =[1 if i in emb else 0 for i in range(n_skills)]
    out[idx]["emb"] = emb 
    out[idx]["description"] = results2[idx].puzzle_description
    out[idx]["interestingness_f"] = results2[idx].interestingness_score_f
    out[idx]["interestingness_g"] = results2[idx].interestingness_score_g
    out[idx]["quality"] = (out[idx]["interestingness_f"]+out[idx]["interestingness_g"])/2
    out[idx]["is_valid"] = results1[idx].puzzle_check.is_valid
    out[idx]["is_valid_explanation"] = results1[idx].puzzle_check.explanations

list_p3 = [P3ProbSolResult(**p) for p in out]
correct_pb=0
for probsol in list_p3:
    if isinstance(probsol.result_obj, ExecResult):
        continue
    if isinstance(probsol.result_obj, str):
        eval_code = (
            f"{probsol.program_str}\n"
            f"def run_eval():\n"
            f"    return f('{probsol.result_obj}')"
        )
    else:
        eval_code = (
            f"{probsol.program_str}\n"
            f"def run_eval():\n"
            f"    return f({probsol.result_obj})"
        )
    # Run code to see if g6_2 solves f6_2
    result = pool_exec_processes(
        eval_code,
        func_name="run_eval",
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
with open(path_embed, "w") as f:
    json.dump(list_program_str, f, indent=4)
# #load it

