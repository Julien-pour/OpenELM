import sys
sys.path.append('/home/flowers/work/OpenELM')
from openelm.environments.p3.p3 import P3ProbSolResult
import json
import pickle
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description="argument parsing")
parser.add_argument("-p", "--base_path", type=str, help="path to maps",default="/home/flowers/work/OpenELM/logs/elm/24-02-05_15:39/step_130/maps.pkl")
args = parser.parse_args()

snapshot_path=args.base_path

with open(snapshot_path, "rb") as f:
    maps = pickle.load(f)

genomes = maps
# fitnesses = maps["fitnesses"]
# genomes = maps["genomes"]
# non_zeros = maps["nonzero"]
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