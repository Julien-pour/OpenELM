# old
from langchain.schema import HumanMessage
from openelm.environments.p3 import label_puzzle_chatgpt
from langchain.chat_models import ChatOpenAI
from joblib import Parallel, delayed
import pickle
import numpy as np
from tqdm import tqdm
import json
path = "/media/data/flowers/OpenELM/tests/subset2label_jul+lae.json"


path_save = "/media/data/flowers/OpenELM/tests/subset2label_jul+lae_cot.json"

n_jobs=6

cfg: dict = {
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.,
    "max_retries":100,
    # TODO: rename config option?
    "model_name": "gpt-3.5-turbo-0613",#"gpt-4-0613",
    "request_timeout": 70
}
chatGPT = ChatOpenAI(**cfg) 
def gen_response(prompt):
    response=chatGPT.generate([[HumanMessage(content=prompt)]])
    return response.generations[0][0].text  


    
def label_puzzle(program_str,n_attempts=0,chatGPT=chatGPT):
    return label_puzzle_chatgpt(chatGPT,program_str,n_attempts=n_attempts,return_completion=True)


with open(path, 'r') as f:
    allitems = json.load(f)
    
list_phenotype_correct_puzzle = Parallel(n_jobs=7)(delayed(label_puzzle)(puzzl["program_str"]) for puzzl in tqdm(allitems))
print(len(list_phenotype_correct_puzzle),len(allitems))
for idx in range(len(list_phenotype_correct_puzzle)):
    allitems[idx]["emb"]=list_phenotype_correct_puzzle[idx][0]
    allitems[idx]["completion"]=list_phenotype_correct_puzzle[idx][1]
with open(path_save, 'w') as f:
    json.dump(allitems,f, indent=4)
