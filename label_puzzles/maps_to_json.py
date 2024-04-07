import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
# /!\ change seed here
path = f"/home/flowers/work/OpenELM/logs/elm/elm_nlp_quality_seed-1/24-04-06_15:10/step_499/maps.pkl"

path_save=path.split("maps.pkl")[0]
with open(path, 'rb') as f:
    maps = pickle.load(f)
genomes = maps["genomes"]
# print(len(genomes.archive),(len(genomes.archive)-140)/(499*50))
puzzles = [i.__dict__ for i in genomes.archive]
list_rm=["result_obj","config","interestingness_f","interestingness_g","is_valid","is_valid_explanation"]
for i in puzzles:
    for key in list_rm:
        if key in i:
            del i[key]
rm_path = path.split("logs/elm/")[1]
step = "step_"+rm_path.split("/step_")[1].split("/maps.pkl")[0]
name = rm_path.split("/")[0] #name_seed_{seed}
full_name= f"{name}_{step}.json" #{name}_seed_{seed}_step_{step}

with open(path_save+full_name, "w") as f:
    json.dump(puzzles, f, indent=4)