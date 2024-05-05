import json
import argparse
import os
import portalocker

parser = argparse.ArgumentParser(description="Example script for argument parsing")
parser.add_argument("-id","--arg_yes", type=int, help=" model",default=0)

args = parser.parse_args()

path_base="/projets/flowers/julien/OpenELM/logs/archives/"#"/home/flowers/work/OpenELM/logs/archives/"#
list_archive=["rd_gen_seed-1.json",
        #    "elm_quality_seed-1.json",
           "elm_nlp_quality_seed-1.json",
           "elm_nlp_seed-1.json",
           "aces_seed-1.json",
           "aces_quality_seed-1.json",
        #    "aces_smart_quality_seed-1.json",
           ]
list_emb= [path_base+archive for archive in list_archive]
bs=4
print("--------------- start generating quality -----------------")
list_synth_prompt=["yes_skills_improvement","yes_gpt4","yes_gpt4_2","yes_mixtral","yes_opus","yes_opus_2"]
path_hf ="/projets/flowers/julien/hf/deepseek-coder-1.3b-instruct"
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_label import ret_list_fitness
tokenizer = AutoTokenizer.from_pretrained(path_hf)
model = AutoModelForCausalLM.from_pretrained(path_hf,device_map='auto')
path_embed = list_emb[args.arg_yes] 
for name_yes in list_synth_prompt:
    print ("path_embed=",path_embed)



    with open(path_embed, "r") as f:
        out = json.load(f)

    list_program_str = [p["program_str"] for p in out]
    list_fitness = ret_list_fitness(tokenizer,model,list_program_str,bs=bs,yes_mode=name_yes)
    for idx in range(len(out)):
        # if quality is a dict, we add the quality
        if isinstance(out[idx]["quality"],dict):
            out[idx]["quality"][name_yes]=list_fitness[idx]
        else:
            dic_qual={name_yes:list_fitness[idx]}
            out[idx]["quality"]=dic_qual


    with open(path_embed, "w") as f:
        json.dump(out, f, indent=4)