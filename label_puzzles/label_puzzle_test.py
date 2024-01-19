from langchain.schema import HumanMessage
from openelm.environments.p3 import skills_evaluation
from langchain.chat_models import ChatOpenAI
from joblib import Parallel, delayed
import pickle
import numpy as np
from tqdm import tqdm
from utils_test import preprocessing_P3_no_test
import json
from openelm.environments.p3 import label_puzzle_chatgpt

testpuzzle = preprocessing_P3_no_test(split = "test", n_token_max = 1024)

path_save = "/home/flowers/work/OpenELM/"+"P3_test"
n_jobs=25

cfg: dict = {
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.,
    "max_retries":100,
    # TODO: rename config option?
    "model_name": "gpt-3.5-turbo-0613",
    "request_timeout": 70
}
chatGPT = ChatOpenAI(**cfg) 
def gen_response(prompt):
    response=chatGPT.generate([[HumanMessage(content=prompt)]])
    return response.generations[0][0].text  


    
def label_puzzle(program_str,n_attempts=0,chatGPT=chatGPT):
    try:
        return label_puzzle_chatgpt(chatGPT,program_str,n_attempts=n_attempts,return_completion=False)
    except:
        return [0. for _ in range(10)]


list_phenotype_correct_puzzle = Parallel(n_jobs=5)(delayed(label_puzzle)(puzzl["program_str"]) for puzzl in tqdm(testpuzzle))
list_phenotype_correct_puzzle_arr= np.array(list_phenotype_correct_puzzle)
 
np.save(path_save+".npy",list_phenotype_correct_puzzle_arr)

for idx_puzz in range(len(list_phenotype_correct_puzzle_arr)):
    testpuzzle[idx_puzz]["emb"]=list_phenotype_correct_puzzle[idx_puzz] 
with open(path_save+".json", 'w') as f:
    json.dump(testpuzzle, f, indent=4)
