import sys
sys.path.append('/home/flowers/work/OpenELM')
from src.openelm.utils.code_eval import pool_exec_processes,return_f,add_return_bool_2_f,return_g,merge_Q_and_A
import tiktoken
import requests
import json
import numpy as np

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
            puzzle_2_add={}
            puzzle_2_add["f"] = add_return_bool_2_f(return_f(i))
            puzzle_2_add["g"] = return_g(i,puzzle_2_add["f"])
            puzzle_2_add['attempts'] = 1 # 
            puzzle_2_add["program_str"] = merge_Q_and_A([(puzzle_2_add["f"],puzzle_2_add["g"])])[0]
            if "List" in puzzle_2_add["program_str"] and not "from typing import List" in puzzle_2_add["program_str"]:
                puzzle_2_add["program_str"]="from typing import List \n"+puzzle_2_add["program_str"]
        
            generated_programs.append(puzzle_2_add["program_str"])
            
            
            results = pool_exec_processes(
                puzzle_2_add["program_str"],
                func_name="g",debug =False,
                processes=10
                )
            puzzle_2_add["result_obj"]=results[0]
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
                
                
                results = pool_exec_processes(
                    puzzle_2_add["program_str"],
                    func_name="run_eval",debug =False,
                    processes=10
                    )
                puzzle_2_add["result_obj"]=None
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
        #remove item where index is False
        puzzles_set = [item for i, item in enumerate(puzzles_set) if index[i]]
        print("number of puzzles after removing too long puzzles",len(puzzles_set))
        
        return puzzles_set
