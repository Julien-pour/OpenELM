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
from openelm.environments.p3 import skills_evaluation
from tqdm import tqdm
import os
from tenacity import *

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
    return chatGPT.generate([[HumanMessage(content=prompt)]])

path_embed = script_dir+"/src/openelm/utils/preprocess_p3_emb.json"
print(script_dir)
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

out = preprocessing_P3(load_embedding = False)

# results = [
#     {"program_str": gen_prog, "result_obj": res_obj, "config": self.config}
#     for (gen_prog, res_obj) in zip(generated_programs, results)
# ]
for i in out:
    del i["f"], i["g"],i["attempts"]
    i["config"] = config.env

out=out
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

