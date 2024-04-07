import sys
sys.path.append('/home/flowers/work/OpenELM')
from src.openelm.utils.code_eval import pool_exec_processes
from utils_label import preprocessing_P3
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.environments.p3.p3 import P3ProbSolResult
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import json
from openelm.environments.p3 import create_prompt_label,Topics_evaluation
from openelm.mutation_model import get_model,get_multiple_completions,get_multiple_completions_instructor
from tqdm import tqdm
import os
from tenacity import *
import instructor

script_dir = os.path.dirname(__file__) 


bs=4
generate_skills=False
generate_description=False
generate_quality=True

n_skills=20 # length of the skill vector
max_workers=40
path_embed = "/home/flowers/work/OpenELM/logs/elm/elm_nlp_seed-1/24-04-05_17:56/step_499/elm_nlp_seed-1_step_499.json"#"/home/flowers/work/OpenELM/label_puzzles/preprocess_p3_emb.json"#script_dir+"/src/openelm/utils/preprocess_p3_emb.json"

with initialize(version_base="1.2"):
    cfg = compose(config_name="elmconfig")
    # print(cfg)
config = OmegaConf.to_object(cfg)

cfg_generation: dict = {
            "temperature": 0.,
            "model": config.model.model_path,
        }


if generate_skills or generate_description:
    client = get_model(config.model)
    instructor_client = instructor.patch(client)


with open(path_embed, "r") as f:
    out = json.load(f)


# preprocess puzzles
if generate_skills:


    batch_prompt1=[create_prompt_label(puzz["program_str"],mode= "give_skills") for puzz in out]
    batch_tools1=[Topics_evaluation for _ in range(len(batch_prompt1))]
    batch_prompt2=[create_prompt_label(puzz["program_str"],mode = "description") for puzz in out]
    # batch_tools2=[Puzzle_Interestingness for _ in range(len(batch_prompt1))]

    print("--------------- start generating labels puzzles -----------------")
    results1 = get_multiple_completions_instructor(client, batch_prompt = batch_prompt1, cfg_generation=cfg_generation, batch_tools= batch_tools1,max_workers=max_workers,temperature=0.0)
    results1_processed = []
    for result in results1:
        skill=result.index_topics
        explanation_skill =result.explanations_index_topics
        if not len(skill)<=5: # we should have at most 5 topics
            skill=skill[:5]
        skill =[1 if i in skill else 0 for i in range(n_skills)]
        # tool_diversity = Puzzle_Diversity
        # tool_interstingness = Puzzle_Interestingness
        dic_label={"emb":skill,"explanation_emb":explanation_skill}
        results1_processed.append(dic_label)



    print("--------------- end label puzzles -----------------")
    for idx in range(len(results1)):  
        out[idx].update(results1_processed[idx])

    with open(path_embed, "w") as f:
        json.dump(out, f, indent=4)


if generate_description:
    print("--------------- start generating description -----------------")

    results2 = get_multiple_completions(client, batch_prompt = batch_prompt2, cfg_generation=cfg_generation,max_workers=max_workers,temperature=0.0)
    print("--------------- end generating description -----------------")

    if len(results1)!=len(results2):
        print(f"len results1 = {len(results1)}, len results2 = {len(results2)}")
        print(" /!\ results1 and results2 should have the same length /!\ ")

    for idx in range(len(results2)):  
        out[idx]["description"] = results2[idx]
        out[idx]["quality"] = 1#(out[idx]["interestingness_f"]+out[idx]["interestingness_g"])/2
        for key in ["interestingness_f","interestingness_g","is_valid","is_valid_explanation"]: # remove old stuff
            if key in out[idx]:
                del out[idx][key]

    with open(path_embed, "w") as f:
        json.dump(out, f, indent=4)



try:
    for i in out:
        config.env.GPT_feedback=True
        i["config"] = config.env
    list_p3 = [P3ProbSolResult(**p) for p in out]
    correct_pb=0
    for probsol in list_p3:
        # if isinstance(probsol.result_obj, ExecResult):
        #     continue
        eval_code = (
            f"{probsol.program_str}\n"
            f"def run_eval():\n"
            f"    return f(g()) == True"
        )
        # Run code to see if g6_2 solves f6_2
        result = pool_exec_processes(
            eval_code,
            func_name="run_eval",
            debug=True
        )
        if result[0] is True:
            correct_pb+=1
        else:
            print("\n--------\nincorrect pb:",eval_code)
            
    print("correct pb", correct_pb)
    print("total_pb",len(list_p3))
except Exception as e:
    print(e)
    print("Error in the evaluation of the puzzles")
    pass

# list_program_str=[p.__to_dict__() for p in list_p3]



if generate_quality:
    print("--------------- start generating quality -----------------")

    path_hf ="/home/flowers/work/hf/deepseek-coder-1.3b-instruct"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils_label import ret_list_fitness
    tokenizer = AutoTokenizer.from_pretrained(path_hf)
    model = AutoModelForCausalLM.from_pretrained(path_hf,device_map='auto')

    list_program_str = [p["program_str"] for p in out]
    list_fitness = ret_list_fitness(tokenizer,model,list_program_str,bs=bs)
    for idx in range(len(out)):
        out[idx]["fitness"]=list_fitness[idx]

    with open(path_embed, "w") as f:
        json.dump(out, f, indent=4)


print("n puzzles=", len(out))
## need to add quality labeling



