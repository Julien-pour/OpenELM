import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
# /!\ change seed here
# path = f"/home/flowers/work/OpenELM/logs/elm/elm_nlp_seed-1/24-04-05_17:56/step_499/maps.pkl"
list_path=["/home/flowers/work/OpenELM/logs/elm_plafrim/aces_quality_seed-1/24-04-13_10:43/step_499/",
           "/home/flowers/work/OpenELM/logs/elm_plafrim/aces_seed-1/24-04-12_16:00/step_499/",
           "/home/flowers/work/OpenELM/logs/elm_plafrim/aces_smart_quality_seed-1/24-04-12_16:22/step_499/",
           "/home/flowers/work/OpenELM/logs/elm_plafrim/elm_nlp_quality_seed-1/24-04-15_15:06/step_499/",
           "/home/flowers/work/OpenELM/logs/elm_plafrim/elm_nlp_seed-1/24-04-13_11:58/step_499/",
           "/home/flowers/work/OpenELM/logs/elm_plafrim/elm_quality_seed-1/24-04-13_18:10/step_499/",
           "/home/flowers/work/OpenELM/logs/elm_plafrim/rd_gen_seed-1/24-04-13_00:36/step_499/",
]
path_save="/home/flowers/work/OpenELM/logs/archives/"
list_path= [path+"maps.pkl" for path in list_path]
for path in list_path:
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
    rm_path = path.split("logs/elm_plafrim/")[1]
    # step = "step_"+rm_path.split("/step_")[1].split("/maps.pkl")[0]
    name = rm_path.split("/")[0] #name_seed_{seed}
    full_name= f"{name}.json" #{name}_seed_{seed}

    with open(path_save+full_name, "w") as f:
        json.dump(puzzles, f, indent=4)