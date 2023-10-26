from langchain.schema import HumanMessage
from openelm.environments.p3 import skills_evaluation
from langchain.chat_models import ChatOpenAI
from joblib import Parallel, delayed
import pickle
import numpy as np
from tqdm import tqdm
from utils_test import preprocessing_P3_no_test
import json
testpuzzle = preprocessing_P3_no_test(split = "test", n_token_max = 1024)

path_save = "/media/data/flowers/OpenELM/"+"_phenotype_test"

n_jobs=5

cfg: dict = {
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 1.,
    "max_retries":100,
    # TODO: rename config option?
    "model_name": "gpt-3.5-turbo-0613",
    "request_timeout": 35
}
chatGPT = ChatOpenAI(**cfg) 
def gen_response(prompt):
    response=chatGPT.generate([[HumanMessage(content=prompt)]])
    return response.generations[0][0].text  


    
def label_puzzle(program_str,n_attempts=0):#,list_prgrm_str=list_prgrm_str,labels_2=labels_2):
    """
    Label a puzzle with the skills it requires
    TODO: add a safeguard if the model hallucinate too much e.g len(category_idx_predicted) > n_skills
    """
    # if program_str in list_prgrm_str:
    #     idx_prgm=list_prgrm_str.index(program_str)
    #     return list(labels_2[idx_prgm])
    prompt,n_skills = skills_evaluation(program_str)
    if n_attempts > 5: # should not append but just in case
        # raise ValueError("too many attempts to label the puzzle")
        print("WARNING: too many attempts to label the puzzle")
        return [0. for i in range(n_skills)]
    response = gen_response(prompt=prompt)
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


list_phenotype_correct_puzzle = Parallel(n_jobs=5)(delayed(label_puzzle)(puzzl["program_str"]) for puzzl in tqdm(testpuzzle))
list_phenotype_correct_puzzle_arr= np.array(list_phenotype_correct_puzzle)
 
np.save(path_save+".npy",list_phenotype_correct_puzzle_arr)

for idx_puzz in range(len(list_phenotype_correct_puzzle_arr)):
    testpuzzle[idx_puzz]["emb"]=list_phenotype_correct_puzzle[idx_puzz] 
with open(path_save+".json", 'w') as f:
    json.dump(testpuzzle, f, indent=4)
