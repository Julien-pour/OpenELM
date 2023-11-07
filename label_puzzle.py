from langchain.schema import HumanMessage
from openelm.environments.p3 import label_puzzle_chatgpt
from langchain.chat_models import ChatOpenAI
from joblib import Parallel, delayed
import pickle
import numpy as np
from tqdm import tqdm
# /!\ change seed here
seed=2
path = f"/media/data/flowers/OpenELM/run_saved/limited_trainset/elm/seed{seed}/step_499/maps.pkl"


path_save = path.split("maps.pkl")[0]+"_phenotype.npy"


n_jobs=15

cfg: dict = {
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.,
    "max_retries":1000,
    # TODO: rename config option?
    "model_name": "gpt-3.5-turbo-0613",#"gpt-4-0613",
    "request_timeout": 70
}
chatGPT = ChatOpenAI(**cfg) 
def gen_response(prompt):
    response=chatGPT.generate([[HumanMessage(content=prompt)]])
    return response.generations[0][0].text  


    
    
def getallitems(maps):
    """
    Returns all the phenotypes that are in the Map."""
    genomes = maps["genomes"]
    valid_phenotype=[]
    for gen in np.ndindex(genomes.shape):
        value_gen = type(genomes[gen])
        if value_gen!=float and value_gen!=int:
            valid_phenotype.append(genomes[gen])
    return valid_phenotype
    

# if need to relabel the last iteration of the elm run (here from step 399 to 499)
# path_2 = "/media/data/flowers/OpenELM/run_saved/elm/23-09-13_23:04/step_499/maps.pkl"

# with open(path_2, 'rb') as f:
#     maps_2 = pickle.load(f)
# path_load_2 = path_2.split("maps.pkl")[0]+"_phenotype.npy"
# allitems_2 = getallitems(maps_2)
# items_gen_2 = [item for item in allitems_2 if item.idx_generation!=-1]
# labels_2=np.load(path_load_2)

# list_prgrm_str=[puz.program_str for puz in items_gen_2]


def label_puzzle(program_str,n_attempts=0,chatGPT=chatGPT):
    return label_puzzle_chatgpt(chatGPT,program_str,n_attempts=n_attempts,return_completion=False)

# def label_puzzle(program_str,n_attempts=0):#,list_prgrm_str=list_prgrm_str,labels_2=labels_2):
#     """
#     Label a puzzle with the skills it requires
#     TODO: add a safeguard if the model hallucinate too much e.g len(category_idx_predicted) > n_skills
#     """
#     # if program_str in list_prgrm_str:
#     #     idx_prgm=list_prgrm_str.index(program_str)
#     #     return list(labels_2[idx_prgm])
#     prompt,n_skills = skills_evaluation(program_str)
#     if n_attempts > 5: # should not append but just in case
#         # raise ValueError("too many attempts to label the puzzle")
#         print("WARNING: too many attempts to label the puzzle")
#         return [0. for i in range(n_skills)]
#     response = gen_response(prompt=prompt)
#     split_completion = response.split("Therefore, the list of indices for the problem is:") # add assert 
#     if len(split_completion) == 2 :#"Skills parsing
#         if split_completion[1][-1] == ".":
#             split_completion[1] = split_completion[1][:-1] 
#         try :
#             category_idx_predicted = eval(split_completion[1]) 
#             list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)]
#             return list_skill
        
#         except: # if pb when parsing try to fix them
#             if split_completion[1].count("]")==1:
#                 try:
#                     category_idx_predicted = eval(split_completion[1].split("]")[0]+"]")
#                     list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)] 
#                     return list_skill
#                 except:
#                     return label_puzzle(program_str,n_attempts=n_attempts+1)
#             else:
#                 return label_puzzle(program_str,n_attempts=n_attempts+1)
        
#     else: 
#         return label_puzzle(program_str,n_attempts=n_attempts+1)


with open(path, 'rb') as f:
    maps = pickle.load(f)



allitems = getallitems(maps)
items_gen = [item for item in allitems if item.idx_generation!=-1]



# list_phenotype_correct_puzzle=[]
# for puzzl in tqdm(list_correct_puzzle[:10]):
#     list_phenotype_correct_puzzle.append(label_puzzle(puzzl.program_str))
#     print(list_phenotype_correct_puzzle)
list_phenotype_correct_puzzle = Parallel(n_jobs=n_jobs)(delayed(label_puzzle)(puzzl.program_str) for puzzl in tqdm(items_gen))
list_phenotype_correct_puzzle= np.array(list_phenotype_correct_puzzle)
 
np.save(path_save,list_phenotype_correct_puzzle)

