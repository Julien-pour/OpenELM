import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
# /!\ change seed here
# path = f"/home/flowers/work/OpenELM/logs/elm/elm_nlp_seed-1/24-04-05_17:56/step_499/maps.pkl"
list_path=["/home/flowers/work/OpenELM/logs/elm_jz/Meta-Llama-3-70B-Instruct-GPTQ/aces_smart_seed-1/24-05-06_05:29/step_120",
            "/home/flowers/work/OpenELM/logs/elm_jz/Meta-Llama-3-70B-Instruct-GPTQ/elm_nlp_seed-1/24-05-06_06:03/step_120",
            "/home/flowers/work/OpenELM/logs/elm_jz/Meta-Llama-3-70B-Instruct-GPTQ/rd_gen_seed-1/24-05-06_06:52/step_120",
            "/home/flowers/work/OpenELM/logs/elm_jz/CodeQwen1.5-7B-Chat/aces_smart_seed-1/24-05-06_05:27/step_300",
            "/home/flowers/work/OpenELM/logs/elm_jz/CodeQwen1.5-7B-Chat/rd_gen_seed-1/24-05-06_06:17/step_300"
]
tmp=True # save in tmp file
list_path= [path+"/maps.pkl" for path in list_path]
for path in list_path:
    if "llama" in path.lower():
        path_save="/home/flowers/work/OpenELM/logs/archives/llama-70/"

    elif "codeqwen"  in path.lower(): 
        path_save="/home/flowers/work/OpenELM/logs/archives/codeqwen/"
    else:
        raise ValueError("model unknow not found")
    if tmp:
        path_save+="tmp/"

    print("converting: ",path)
    # path_save=path.split("maps.pkl")[0]
    with open(path, 'rb') as f:
        maps = pickle.load(f)
    genomes = maps["genomes"]
    # print(len(genomes.archive),(len(genomes.archive)-140)/(499*50))
    puzzles = [i.__dict__ for i in genomes.archive]
    list_rm=["result_obj","config","interestingness_f","interestingness_g","is_valid","is_valid_explanation"]
    for i in puzzles:
        if isinstance(i["emb"],np.ndarray):
            i["emb"]=i["emb"].tolist()

        for key in list_rm:
            if key in i:
                del i[key]
    rm_path = path.split("logs/elm_jz/")[1]
    # step = "step_"+rm_path.split("/step_")[1].split("/maps.pkl")[0]
    name = rm_path.split("/")[1] #name_seed_{seed}
    full_name= f"{name}.json" #{name}_seed_{seed}

    with open(path_save+full_name, "w") as f:
        json.dump(puzzles, f, indent=4)