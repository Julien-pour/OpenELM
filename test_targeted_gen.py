from src.openelm.utils.code_eval import preprocessing_P3,pool_exec_processes
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.environments.p3.p3 import P3ProbSolResult
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import pickle
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from tenacity import *
import numpy as np
from openelm.environments.p3 import P3_probsol_chat_med_seed,skills_evaluation,P3_probsol_chat_med_seed_goal_targeted,P3_IMPORTS

n_jobs=5
cfg: dict = {
    "max_tokens": 1024,
    "temperature": 0.9,
    "top_p": 0.95,
    # TODO: rename config option?
    "model_name": "gpt-3.5-turbo-0613",
}

chatGPT = ChatOpenAI(**cfg)    
script_dir = os.path.dirname(__file__) 

# @retry(stop=stop_after_attempt(10),wait=wait_random_exponential(min=1, max=40))
def gen_response(prompt):
    out=chatGPT.generate([[HumanMessage(content=prompt)]])

path = "/media/data/flowers/OpenELM/logs/elm/23-08-21_18:50/step_9/save_all.json"
import copy

def extract_puzzle(responses):
    list_pb=[]
    # parse the generated code 
    for gen_prog in responses:
        split_pb = copy.deepcopy(gen_prog.replace("```python","```").replace("``` python","```").replace("```\n","```").split("```"))
        for idx in range(len(split_pb)):
            if "def f" in split_pb[idx] and "def g" in split_pb[idx]:
                list_pb.append(split_pb[idx])
    for idx_assert in range(len(list_pb)):
    #     list_pb[idx] = list_pb[idx].split("assert")[0]+"assert f(g()) == True"
        if not "assert f(" in list_pb[idx_assert]:
            list_pb[idx_assert] = list_pb[idx_assert] + "\nassert f(g()) == True"
    generated_programs = list_pb
    
    list_lib = ["math", "random", "itertools"]
    
    for idx in range(len(generated_programs)):
        if not P3_IMPORTS in generated_programs[idx]:
            generated_programs[idx] = P3_IMPORTS+ generated_programs[idx]
            
        # check if lib are correctly imported (if not import them)
        for lib in list_lib:
            if lib in generated_programs[idx]:
                if not f"import {lib}" in  generated_programs[idx].split("def f")[0]:
                    generated_programs[idx] = f"import {lib}\n" + generated_programs[idx]

def label_puzzle(program_str,n_attempts=0):
    """
    Label a puzzle with the skills it requires"""
    prompt,n_skills = skills_evaluation(program_str)
    if n_attempts > 4: # should not append but just in case
        return [0. for i in range(n_skills)]
    
    response = gen_response(prompt)
    response = response.generations[0][0].text    
    split_completion = response.split("Therefore, the list of indices for the problem is:") # add assert 
    if len(split_completion) == 2 :#"Skills parsing
        if split_completion[1][-1] == ".":
            split_completion[1] = split_completion[1][:-1] 
        try :
            category_idx_predicted = eval(split_completion[1]) 
            list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)]
            return list_skill
        
        except: # if pb when parsing try to fix them
            if split_completion[1].count("]")==1:
                try:
                    category_idx_predicted = eval(split_completion[1].split("]")[0]+"]")
                    list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)] 
                    return list_skill
                except:
                    return label_puzzle(program_str,n_attempts=n_attempts+1)
            else:
                return label_puzzle(program_str,n_attempts=n_attempts+1)
        
    else: 
        return label_puzzle(program_str,n_attempts=n_attempts+1)

with initialize(version_base="1.2"):
    cfg = compose(config_name="elmconfig")
    # print(cfg)
config = OmegaConf.to_object(cfg)

path = "/media/data/flowers/OpenELM/logs/elm/23-08-21_18:50/step_9/save_all.json"
with open(path, 'r') as f:
    data = json.load(f)

class P3Solution:
    def __init__(self,program_str,emb):
        """
        Genotype for a programming puzzle solution.
        Args:
            program_str: the solution program string (the g6() function).
            result_obj: dict.
            config: environment config
        """
        self.program_str = program_str
        self.emb = emb
archive_puzz=[]
for puzz in data:
    archive_puzz.append(P3Solution(puzz["program_str"],puzz["emb"]))

list_fullprompt=[]
for _ in range(10):
    skill_targeted = np.random.randint(0, 2, 10).tolist()
    llist_pb =  list(np.random.choice(archive_puzz,size=3))
    prompt = P3_probsol_chat_med_seed_goal_targeted(llist_pb,skill_targeted)
    response = gen_response(prompt)
    resp_str = response.generations[0][0].text
    # list_fullprompt
    
# compute embedding in NLP space
results = Parallel(n_jobs=n_jobs)(delayed(label_puzzle)(puzz["program_str"]) for puzz in tqdm(out))
for i,r in enumerate(results):
    out[i]["emb"] = r


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

list_program_str=[{"program_str" : p.program_str,"emb" : p.emb} for p in list_p3]

# path = "/media/data/flowers/OpenELM/preprocess_p3.json"
with open(path_embed, "w") as f:
    json.dump(list_program_str, f, indent=4)
# #load it

