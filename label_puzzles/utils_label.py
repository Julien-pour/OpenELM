import sys
sys.path.append('/home/flowers/work/OpenELM')
from src.openelm.utils.code_eval import pool_exec_processes,return_f,add_return_bool_2_f,return_g,merge_Q_and_A
import tiktoken
import requests
import json
import numpy as np
import torch
from openelm.quality_metrics.yes import return_proba_yes, return_yes_prompt, return_prompt_format




def prompt_format(text):
    """
    return the prompt format for the model system,user,...
    """
    return return_prompt_format("deepseek-coder", text)


def generate_quality(tokenizer,model,list_text: list[str],batch_size_quality=16):
    soft = torch.nn.Softmax(dim=1)
    assert isinstance(list_text,list)
    with torch.inference_mode():
        list_proba_yes=[]
        for i in range(0, len(list_text), batch_size_quality):
            batch_texts = list_text[i:i+batch_size_quality]
            inputs = tokenizer(batch_texts, return_tensors="pt",padding=True).to("cuda") #maybe need to batch that
            out_yes = model(**inputs)
            # out = self.tokenizer.decode(out_tok[0])
            k=25# get top 25 tokens
            yes_logits=soft(out_yes.logits[:,-1]).cpu().detach() #logits associated with the token "yes"
            values,indices=torch.topk(yes_logits, k)
            list_words=tokenizer.batch_decode(indices.flatten())
            list_words=np.array(list_words).reshape(values.shape).tolist()
            values = values.tolist()
            
            # values,list_token
            for idx in range(len(list_words)):
                # if self.debug:
                #     print("-----")
                #     for j in range(len(list_words[idx])):
                #         print(f"list_words[idx][j]: {list_words[idx][j]}, values[idx][j]: {values[idx][j]}")
                list_proba_yes.append(return_proba_yes(values[idx],list_words[idx]))
    return list_proba_yes


def absolute_grade(tokenizer,model,list_text: list[str],bs):
    """return the absolute_grade float between 0 and 10"""
    assert isinstance(list_text,list)
    yes_mode = "skills_improvement" #TODO: add to config 
    yes_prompt = return_yes_prompt(yes_mode)
    for idx in range(len(list_text)):
        list_text[idx] = prompt_format(yes_prompt.format(datapoint=list_text[idx]))

    out = generate_quality(tokenizer,model,list_text,batch_size_quality=bs) # remove [0] when main loop is batchable
    return out

def ret_list_fitness(tokenizer,model,list_program_str,bs) -> list[float]:

    # check the docstring works fine
    fitness = absolute_grade(tokenizer,model,list_program_str,bs=bs)
    return fitness



def raw_puzzle_to_clean_puzzle(puzzle: dict) -> dict:
    """
    return a clean puzzle from a raw puzzle
    """
    puzzle_2_add={}
    puzzle_2_add["f"] = add_return_bool_2_f(return_f(puzzle))
    puzzle_2_add["g"] = return_g(puzzle,puzzle_2_add["f"])
    puzzle_2_add['attempts'] = 1 # 
    puzzle_2_add["program_str"] = merge_Q_and_A([(puzzle_2_add["f"],puzzle_2_add["g"])])[0]
    if "List" in puzzle_2_add["program_str"] and not "from typing import List" in puzzle_2_add["program_str"]:
        puzzle_2_add["program_str"]="from typing import List \n"+puzzle_2_add["program_str"]
    return puzzle_2_add

def preprocessing_P3(split: str = "train", n_token_max: int =512,debug=False,path_puzzle=None) -> list[dict]:
    """
    dl puzzles from P3 dataset and give train or test puzzles
    split = "train" or "test"
    """
    
    import sys 
    sys.set_int_max_str_digits(10_000)
    if path_puzzle is not None:
        puzzles = json.load(open(path_puzzle))
    else:
        puzzles = requests.get(
            "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
        ).json()
        data_split = requests.get(
            "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/main/puzzles/split.json"
        ).json()
    enc = tiktoken.encoding_for_model("gpt-4")
    puzzles_set=[]
    generated_programs=[]
    count_puzzle_without_solution=0
    for i in puzzles:
        if path_puzzle is not None:
            if i["sol_bodies"]==[]: # no solution
                count_puzzle_without_solution+=1
            puzzle_2_add=raw_puzzle_to_clean_puzzle(i)        
            generated_programs.append(puzzle_2_add["program_str"])
            
            
            # results = pool_exec_processes(
            #     puzzle_2_add["program_str"],
            #     func_name="g",debug =False,
            #     processes=10
            #     )
            # puzzle_2_add["result_obj"]=results[0]
            if i["sol_bodies"]!=[]:
                puzzles_set.append(puzzle_2_add)
        else:
            if i["name"][:-2] in data_split[split] and i["sol_bodies"]!=[]:
                puzzle_2_add={}
                puzzle_2_add["f"] = add_return_bool_2_f(return_f(i))
                puzzle_2_add["g"] = return_g(i,puzzle_2_add["f"])
                puzzle_2_add['attempts'] = 1 # 
                puzzle_2_add["program_str"] = merge_Q_and_A([(puzzle_2_add["f"],puzzle_2_add["g"])])[0]
                generated_programs.append(puzzle_2_add["program_str"])
                
                
                # results = pool_exec_processes(
                #     puzzle_2_add["program_str"],
                #     func_name="run_eval",debug =False,
                #     processes=10
                #     )
                # puzzle_2_add["result_obj"]=None
                puzzles_set.append(puzzle_2_add)
    if path_puzzle is not None:
        print("number of puzzles without solution",count_puzzle_without_solution)
    if split == "test":
        return puzzles_set
    else:
        # remove puzzles that are too long
        List_len_embedding = []
        for puzz in puzzles_set:
            List_len_embedding.append(len(enc.encode(puzz["program_str"])))
        index=np.array(List_len_embedding)<=n_token_max
        print("number of puzzles before removing too long puzzles",len(puzzles_set))
        print("number of puzzles too long",len(puzzles_set)-sum(index))
        for i in range(len(index)):
            if not index[i]:
                print(List_len_embedding[i])
        #remove item where index is False
        puzzles_set = [item for i, item in enumerate(puzzles_set) if index[i]]
        print("number of puzzles after removing too long puzzles",len(puzzles_set))
        
        return puzzles_set
