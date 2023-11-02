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

#     Label a puzzle with the skills it requires
#     TODO: add a safeguard if the model hallucinate too much e.g len(category_idx_predicted) > n_skills
#     """
#     # if program_str in list_prgrm_str:
#     #     idx_prgm=list_prgrm_str.index(program_str)
#     #     return list(labels_2[idx_prgm])
#     prompt,n_skills = skills_evaluation(program_str)
#     if n_attempts > 2: # should not append but just in case
#         # raise ValueError("too many attempts to label the puzzle")
#         print("WARNING: too many attempts to label the puzzle")
#         if return_completion:
#             return [0. for i in range(n_skills)],save_completion
#         else:
#             return [0. for i in range(n_skills)]
#     response = gen_response(prompt=prompt)
#     print(response)
#     save_completion[str(n_attempts)]=response
#     split_completion = response.split("he list of indices for the problem is:") #Therefore, the list of indices for the problem is: 
#     if len(split_completion) == 2 :#"Skills parsing
#         if split_completion[1][-1] == ".":
#             split_completion[1] = split_completion[1][:-1] 
#         try :
#             category_idx_predicted = eval(split_completion[1]) 
#             list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)]
#             save_completion[str(n_attempts)]=[response,list_skill]
#             if return_completion:
#                 return list_skill,save_completion
#             else:
#                 return list_skill
        
#         except: # if pb when parsing try to fix them
#             if split_completion[1].count("]")==1:
#                 try:
#                     category_idx_predicted = eval(split_completion[1].split("]")[0]+"]")
#                     list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)] 
#                     save_completion[str(n_attempts)]=[response,list_skill]
#                     if return_completion:
#                         return list_skill,save_completion
#                     else:
#                         return list_skill
#                 except:
#                     # return label_puzzle(program_str,n_attempts=n_attempts+1)
#                     pass
#             # else:
#             #     return label_puzzle(program_str,n_attempts=n_attempts+1)
        
#     if return_completion:
#         return label_puzzle(program_str,n_attempts=n_attempts+1,save_completion=save_completion)
#     else:
#         return label_puzzle(program_str,n_attempts=n_attempts+1)


with open(path, 'r') as f:
    allitems = json.load(f)
    
list_phenotype_correct_puzzle = Parallel(n_jobs=7)(delayed(label_puzzle)(puzzl["program_str"]) for puzzl in tqdm(allitems))
print(len(list_phenotype_correct_puzzle),len(allitems))
for idx in range(len(list_phenotype_correct_puzzle)):
    allitems[idx]["emb"]=list_phenotype_correct_puzzle[idx][0]
    allitems[idx]["completion"]=list_phenotype_correct_puzzle[idx][1]
with open(path_save, 'w') as f:
    json.dump(allitems,f, indent=4)
