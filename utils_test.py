import numpy as np
from pebble import ProcessPool

import ast
import copy
import tiktoken
import json
def test_puzzle(test_fg):
    test_fg= "from typing import *\n"+test_fg
    try:
        exec(test_fg)
        return True,test_fg
    except Exception as e:
        # print(str(e))
        # print("program not working: "+test_fg)
        return False,test_fg

def judge_parallel(src_codes, timeout=10., max_workers=10):

    max_workers = min(len(src_codes), max_workers)

    codes = src_codes
    successes = set()
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(test_puzzle, [code for code in codes], timeout=timeout)

        results = future.result()
        i = 0
        while True:
            try:
                success, code = next(results)
                if success:
                    successes.add(codes[i])
            except StopIteration:
                break
            except (TimeoutError, Exception) as error:
                pass
            assert i < len(codes)
            i += 1
        assert i == len(codes)
    # utils.silence_std_err(False)
    return [code in successes for code in src_codes]


prompt_solve_puzzle='''You will be given a function and its docstring. Respond only in code with a correct, efficient implementation of the function.
You need to generate the correct solutions (g), for the Problem 3 that satisfies the condition f(g()) == True.
Problem 0:
```
def f(stamps: List[int], target=80, max_stamps=4, options=[10, 32, 8]) -> bool:
    """Find a selection of at most max_stamps stamps whose total worth is the target value."""
    for s in stamps:
        assert s in options
    return len(stamps) <= max_stamps and sum(stamps) == target
```
Solution 0:
```
def g(target = 80, max_stamps = 4, options = [10, 32, 8]):
    from itertools import combinations_with_replacement
    for n in range(max_stamps + 1):
        for c in combinations_with_replacement(options, n):
            if sum(c) == target:
                return list(c)
assert f(g())
```
Problem 1:
```
from typing import*
def f(ans: List[List[int]], target=2) -> bool:
    """
    Find a list of pairs of integers where the number of pairs in which the second number is more than
    two greater than the first number is a given constant
    """
    for i in range(len(ans)):
        a, b = ans[i]
        if b - a >= 2:
            target -= 1
    return target == 0
```
Solution 1:
```
def g(target = 2):
    return [[0, 2]] * target 
assert f(g()) == True
```
Problem 2:
```
def f(n: int, v=313946483, w=806690290) -> bool:
    """Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w."""
    for i in range(n):
        assert v <= w
        v *= 3
        w *= 2
    return v > w
```
Solution 2:
```
def g(v = 313946483, w = 806690290):
    i = 0
    while v <= w:
        v *= 3
        w *= 2
        i += 1
    return i 
assert f(g()) == True
```
Problem 3:
```
{pb}
```
Solution 3:
```
{g_firstline}
'''


def pass_at_k(n, c, k):
    """
    Adapted from "Evaluating Large Language Models Trained on Code" (https://arxiv.org/abs/2107.03374)

    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    assert n >= k
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def return_f(puzzle_json):
    puzzle_json = copy.deepcopy(puzzle_json)
    f = puzzle_json["sat"]
    #  add 'sol_docstring' (description of the problem) to the function f
    f = f.replace("sat(", "f(")
    idx_add_problem_description = f.find("\n")

    if type(puzzle_json["sol_docstring"]) == str:
        f=f[:idx_add_problem_description+1]+ puzzle_json["sol_docstring"]+"\n"+f[idx_add_problem_description+1:]
    return f

def extract_args_f(f):
    """
    extract arguments of f, for g
    """
    str_arg=""
    parsed_ast = ast.parse(f)
    func=parsed_ast.body[0]
    name_args = [a.arg for a in func.args.args][1:] # remove the first arg as it isn't necessary for g (because it is the output return by g)
    assert len(func.args.defaults) == len(name_args)
    for i in range(len(name_args)):
        def_values = ast.literal_eval(func.args.defaults[i])
        if type(def_values) == str:
            def_values = "'"+def_values+"'"
        str_arg += name_args[i] + " = " + str(def_values)
        if i < len(name_args)-1:
            str_arg+=", "
    return str_arg

def add_return_bool_2_f(f):
    tree = ast.parse(f)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.returns = ast.Name(id='bool', ctx=ast.Load())

    return ast.unparse(tree)


def return_header_g(f):
    args_f = extract_args_f(f)
    return "def g("+args_f+"):"
    
def return_g(puzzle_json,f):
    if puzzle_json["sol_bodies"] == []:
        print("no solution in json")
        return "def g(""):\n    pass"
    args_f = extract_args_f(f)
    g = "def g("+args_f+"):\n"+copy.deepcopy(puzzle_json["sol_bodies"])[0]
    return g

def merge_Q_and_A(liste_fg):
    parsed = copy.deepcopy(liste_fg) # format [(f,g),(f,g),...]

    judge_srcs = [f"{f}\n{g}\nassert f(g())" for (f, g) in parsed] # format the code to be judged
    return judge_srcs


def preprocessing_P3_no_test(split: str = "train", n_token_max: int = 512, load_embedding = False,debug=False) -> list[dict]:
    """
    dl puzzles from P3 dataset and give train or test puzzles
    split = "train" or "test"
    """
    import os
    # os.environ['HF_DATASETS_CACHE'] = "/projets/flowers/julien/hf/datasets"
    # os.environ['TRANSFORMERS_CACHE'] = "/projets/flowers/julien/models/"
    # from transformers import AutoTokenizer
    # model_id="codellama/CodeLlama-7b-Python-hf"
    # # tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    import sys 
    sys.set_int_max_str_digits(10_000)
    with open("/media/data/flowers/OpenELM/puzzles.json",mode='r') as f:
        puzzles = json.load(f)
    with open("/media/data/flowers/OpenELM/split.json",mode='r') as f:
        data_split = json.load(f)
    
    
    puzzles_set=[]
    generated_programs=[]
    for i in puzzles:
        if i["name"][:-2] in data_split[split] and i["sol_bodies"]!=[]:
            puzzle_2_add={}
            puzzle_2_add["f"] = add_return_bool_2_f(return_f(i))
            puzzle_2_add["g"] = return_g(i,puzzle_2_add["f"]) 
            puzzle_2_add["program_str"] = merge_Q_and_A([(puzzle_2_add["f"],puzzle_2_add["g"])])[0]
            puzzle_2_add["g_firstline"]= return_header_g(puzzle_2_add["f"])
            generated_programs.append(puzzle_2_add["program_str"])
            puzzles_set.append(puzzle_2_add)
    
    enc = tiktoken.encoding_for_model("gpt-4")
    List_len_embedding = []
    for puzz in puzzles_set:
        len_puzz=len(enc.encode(puzz["program_str"]))
        # print(len_puzz)
        List_len_embedding.append(len_puzz)
    index=np.array(List_len_embedding)<=n_token_max
    #remove item where index is False
    puzzles_set = [item for i, item in enumerate(puzzles_set) if index[i]]
    print("puzzle found =",len(puzzles_set))
    return puzzles_set