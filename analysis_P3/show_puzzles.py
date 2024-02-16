import pickle 
import argparse
import numpy as np
import json



parser = argparse.ArgumentParser(description="argument parsing")
parser.add_argument("-p", "--base_path", type=str, help="path to maps",default="/home/flowers/work/OpenELM/analysis_P3/quality/to_analyse/ACES_puzzles.json")
parser.add_argument("-s", "--sample_max", type=str, help="sample for each generation",default=10)
parser.add_argument("-g", "--show_N_gen", type=str, help="how many generation to show",default=30)



args = parser.parse_args()

# load maps
snapshot_path = args.base_path
sample_max = args.sample_max
if "json" in snapshot_path:
    file_mode = "json"
else:
    file_mode =  "pkl"

if file_mode == "json":
    with open(snapshot_path, "r") as f:
        genomes = json.load(f)
else:
    with open(snapshot_path, "rb") as f:
        maps = pickle.load(f)
    fitnesses = maps["fitnesses"]
    genomes = maps["genomes"]
    non_zeros = maps["nonzero"]
    genomes=[i.__dict__ for i in genomes.archive]




def exctract_subset_order_by_gen(data,sample_max=10,show_N_gen=10):

    list_idx=[i["idx_generation"] for i in data]
    gen_max=np.max(list_idx)
    list_data_filtered=[]
    show_N_gen=min(show_N_gen,gen_max)
    gen2sample = list(np.arange(1,gen_max+1,gen_max//show_N_gen))
    if not(gen_max in gen2sample):
        gen2sample.append(gen_max)
# Filtering and sampling
    list_data_filtered = []
    for gen in gen2sample:
        # Filter data for each generation
        filtered_data = [i for i in data if i["idx_generation"] == gen]
        print(len(filtered_data))
        # Sample data if more than 'sample_max' items are available
        if len(filtered_data) > sample_max:
            sampled_data = np.random.choice(filtered_data, sample_max, replace=False)
        else:
            sampled_data = filtered_data
        
        list_data_filtered.extend(sampled_data)
    return list_data_filtered



data_subsample = exctract_subset_order_by_gen(genomes,sample_max)

# len(data_subsample)
list_idx = [i["idx_generation"] for i in genomes]
gen_max = np.max(list_idx)

for puz in data_subsample:
    print(f"\n\n  ========= generation: {puz['idx_generation']} / {gen_max}  fitness: {puz['fitness']}")
    print(puz["program_str"])

print(f"Number of puzzles: {len(genomes)}")