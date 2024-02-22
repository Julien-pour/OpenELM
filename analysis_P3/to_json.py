import sys
sys.path.append('/home/flowers/work/OpenELM')
from openelm.environments.p3.p3 import P3ProbSolResult
import json
import pickle
import argparse
from tqdm import tqdm
# parser = argparse.ArgumentParser(description="argument parsing")
# parser.add_argument("-p", "--base_path", type=str, help="path to maps",default="/home/flowers/work/OpenELM/logs/elm/env=p3_probsol_Chat_IMGEP_smart/24-02-16_16:55/step_90/maps.pkl")
# /home/flowers/work/OpenELM/logs/elm/env=P3ProbSolChatEnv_ELM_NLP/24-02-15_22:57/step_50/maps.pkl
# /home/flowers/work/OpenELM/logs/elm/env=p3_probsol_Chat_IMGEP_smart/24-02-15_22:58/step_50/maps.pkl
# args = parser.parse_args()

# snapshot_path=args.base_path

base_path_snap="/home/flowers/work/OpenELM/analysis_P3/quality/to_analyse/"
seed = "1"
path_aces = "/home/flowers/work/OpenELM/logs/elm/env=p3_probsol_Chat_IMGEP_smart/24-02-16_16:55/step_90"
path_elm_nlp = "/home/flowers/work/OpenELM/logs/elm/env=P3ProbSolChatEnv_ELM_NLP/24-02-16_16:54/step_90"
path_rd_gen = "/home/flowers/work/OpenELM/logs/elm/env=p3_probsol_Chat,qd=mapelites_rd/24-02-16_16:10/step_90"

list_snapshots = [path_aces,path_elm_nlp,path_rd_gen]


def load_maps(snapshot_path):
    if not "pkl" in snapshot_path:
        print("not found")
        if snapshot_path[-1]=="/":
            snapshot_path += "maps.pkl"
        else:
            snapshot_path += "/maps.pkl"


    with open(snapshot_path, "rb") as f:
        maps = pickle.load(f)
    fitnesses = maps["fitnesses"]
    genomes = maps["genomes"]
    non_zeros = maps["nonzero"]
    return genomes,snapshot_path

def to_json(snapshot_path):
    genomes,snapshot_path = load_maps(snapshot_path)
    to_remove =["config","interestingness_f","interestingness_g","result_obj"]
    list_puzzles = []
    for puz in tqdm(genomes):#genomes.archive):
        puzzle_dic = puz.__dict__
        for i in to_remove:
            if i in puzzle_dic:
                del puzzle_dic[i]
        if hasattr(puz, "fitnessPP"):
            puzzle_dic["fitnessPP"] = puz.fitnessPP
        if hasattr(puz, "passk"):
            puzzle_dic["pass_k"] = puz.passk
        list_puzzles.append(puzzle_dic)


    with open(snapshot_path.replace("maps.pkl","puzzles.json"), "w") as f:
        json.dump(list_puzzles, f, indent=4)

    #list_name=["rd_gen","elm","elm_NLP","imgep_random","imgep_smart"]

    if "env=p3_probsol_Chat,qd=mapelites_rd" in snapshot_path:
        filename = "maps_"+seed+"_rd_gen.json"
    elif "p3_probsol_Chat_IMGEP_smart" in snapshot_path:
        filename ="maps_"+seed+"_imgep_smart.json"
    elif "env=P3ProbSolChatEnv_ELM_NLP" in snapshot_path:
        filename ="maps_"+seed+"_elm_NLP.json"
    else:
        raise ValueError("Unknown snapshot path "+snapshot_path)
    with open(base_path_snap + filename, "w") as f:
        json.dump(list_puzzles, f, indent=4)
    base_eval= "/home/flowers/work/evaluate_model/run_saved/"
    with open(base_eval + filename, "w") as f:
        json.dump(list_puzzles, f, indent=4)

for path in list_snapshots:
    to_json(path)
    print("done")