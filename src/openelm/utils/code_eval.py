import functools
import itertools
from scipy.spatial.distance import cdist
import multiprocessing as mp
from typing import Any, Iterable, Optional, Union
import copy
import tiktoken
import ast
import numpy as np
import requests
from openelm.sandbox.server.sandbox_codex_execute import ExecResult, unsafe_execute
import json
import os

import re

def just_remove_example_in_docstring(source_code: str) -> str:
    puzzle_formated= source_code

    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Extract the docstring from function f and remove it
    f_docstring = None
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == 'f':
            if ast.get_docstring(item):
                f_docstring = ast.get_docstring(item)
                if (f_docstring != None):
                    delimiters ="example","Example","For example","Example:"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    f_docstring_split = re.split(regex_pattern, f_docstring)[0]
                    item.body[0].value.s = f_docstring_split
    if (f_docstring != None):
        # Convert the modified AST back to source code
        puzzle_formated=ast.unparse(tree)
    puzzle_formated=puzzle_formated.replace('""""""',"")
    puzzle_formated = os.linesep.join([s for s in puzzle_formated.splitlines() if s.strip()]) # remove empty line

    return puzzle_formated
# def just_remove_example_in_docstring(source_code: str) -> str: # remove all docstring, oh no wrong copy paste
#     puzzle_formated= source_code

#     # Parse the source code into an AST
#     tree = ast.parse(source_code)

#     # Extract the docstring from function f and remove it
#     f_docstring = None
#     for item in tree.body:
#         if isinstance(item, ast.FunctionDef) and item.name == 'f':
#             if ast.get_docstring(item):
#                 f_docstring = ast.get_docstring(item)
#                 item.body[0].value.s = ""
#     if (f_docstring != None):
#         # Convert the modified AST back to source code
#         puzzle_formated=ast.unparse(tree)
#     puzzle_formated=puzzle_formated.replace('""""""',"")
#     puzzle_formated = os.linesep.join([s for s in puzzle_formated.splitlines() if s.strip()]) # remove empty line

#     return puzzle_formated

def remove_docstring(source_code: str) -> str:
    puzzle_formated= source_code

    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Extract the docstring from function f and remove it
    f_docstring = None
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == 'f':
            if ast.get_docstring(item):
                f_docstring = ast.get_docstring(item)
                if (f_docstring != None):
                    item.body[0].value.s = ""
        # Convert the modified AST back to source code
    if (f_docstring != None):
        puzzle_formated=ast.unparse(tree)
    puzzle_formated=puzzle_formated.replace('""""""',"")
    puzzle_formated = os.linesep.join([s for s in puzzle_formated.splitlines() if s.strip()]) # remove empty line

    return puzzle_formated

def pool_exec_processes(
    prompts: Union[str, Iterable[str]],
    func_name: Optional[str] = None,
    args: Optional[dict[str, Any]] = None,
    ground_truth: Optional[dict[tuple, Any]] = None,
    timeout: float = 5.0,
    processes: int = 1,
    debug: bool = False,
) -> list[Any]:
    """
    Execute code in separate process(s).

    Args:
        prompts (str or Iterable): Prompt string(s) to execute.
        func_name (str): Name of function in prompt string to execute.
        args (dict): Arguments to pass to function.
        ground_truth (dict): Dict with args as keys and correct return values as
        values.
        timeout (float): Timeout limit in seconds.
        processes (int): Number of processes to use.
        debug (bool): Whether to print debug messages.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts_2_test=[]
    for i in prompts:
        prompts_2_test.append("\nfrom typing import*\n"+ i) # overkill need to check usefull imports

    eval_fn = functools.partial(
        unsafe_execute,
        func_name=func_name,
        args=args,
        ground_truth=ground_truth,
        timeout=timeout,
        debug=debug,
    )
    if processes <= 1:
        return list(map(eval_fn, prompts_2_test))
    with mp.Pool(processes=processes) as pool:
        results = list(pool.map(eval_fn, prompts_2_test)) # timeout here bug: too much process?
    if debug:
        print(results)
    return results


def eval_completions(
    eval_results: Union[str, Iterable[str]],
    task: str = "parity",
    timeout: float = 5.0,
    processes: int = 1,
    debug: bool = False,
) -> list[Union[int, ExecResult]]:
    """
    Evaluate (a batch of) the modified eval_results on a task.

    Args:
        eval_results: either a string or a batch of strings. The code(s) to be evaluated.
        task: (Optional) the task to be performed.
        timeout: (Optional) the timeout (in seconds).
        processes: (Optional) the number of processes to use.

    Returns:
        A list of status eval_results of the batch of strings.
    """
    if task == "parity":
        if isinstance(eval_results, str):
            eval_results = [eval_results]
        results = pool_exec_processes(
            eval_results,
            func_name="parity",
            ground_truth=parity_test_data,
            timeout=timeout,
            processes=processes,
            debug=debug,
        )
        return results
    else:
        raise ValueError(f"Unknown task: {task}")



def mutate_code(
    n_bugs: int = 5, task: str = "parity", mutate_method="prompt"
) -> tuple[str, str]:
    """
    Mutate code to create n bugs. Output the prompt in diff format.

    Args:
        n_bugs: number of bugs to introduce (from 1 to 5).
        task: (Optional) the task to be performed.
        mutate_method: (Optional) 'diff' or 'prompt',
        corresponding to diff mutation or prompt mutation.

    Returns:
        mutated_code, function_string
    """
    mutation_templates = {
        "diff": [
            f"<NME> {task}.py\n<BEF> ",
            "",  # placeholder for the context, e.g., the buggy code
            "\n<MSG> Fixed bugs",
        ],
        "prompt": [
            "# A buggy implementation\n#!/usr/bin/python3\n",
            "",  # placeholder for the context, e.g., the buggy code
            "\n# Fixed bugs\ndef",
        ],
    }
    mutation_template = mutation_templates[mutate_method]
    if task == "parity":
        variables = ["b", "b", "b", "b", 2]
        for i in range(n_bugs):
            variables[i] = "c" if i < 4 else 3
        func_str = (
            'def parity(b1,b2,b3,b4):\n    """Return binary parity of a sequence of input bits.'
            ' Return 0 for even parity, 1 for odd parity."""\n    bit_sum = sum(['
            "{}1,{}2,{}3,{}4])\n    return bit_sum % {}".format(*variables)
        )
        mutation_template[1] = func_str
        return "".join(mutation_template), func_str
    else:
        raise ValueError(f"Unknown task: {task}")


def parity_reference(b1, b2, b3, b4):
    """
    Return binary parity of a sequence of input bits.

    Return 0 for even parity, 1 for odd parity.
    """
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2


parity_test_data = {
    i: parity_reference(*i) for i in itertools.product(range(2), repeat=4)
}


def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a * x**2 + b * x + c


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


def get_inputs(sat: str):
    """Extacts arguments past the first from a function string
    def f(a: Dict[int, str], b=12):
       test

    should give 'b=12'
    """
    sat = sat.replace(" -> bool", "")
    first_line = sat.split("\n")[0].strip()
    if not(first_line.startswith("def")): #good idea to handle problem if parsing fails
        sat = move_import_inside_function(sat)
        first_line = sat.split("\n")[0].strip()

    if not first_line.endswith("):") and "#" in first_line:
        if "):" in first_line:
            n = first_line.index("):")
            if "#"  in first_line[n:]:
                first_line = first_line[:n + first_line[n:].index("#")].strip()
        else:
            first_line = "" # raises exception below
    if not (first_line.endswith("):") and first_line.startswith("def")):
        print("====================== /!\  Warning  /!\=====================")
        print("Weird puzzle, cannot extract inputs", json.dumps(sat))
        print("====================== /!\  Warning  /!\=====================")
        # raise WeirdInputsException("Weird puzzle, cannot extract inputs", json.dumps(sat))        
    arg_str = first_line[first_line.index("("):-len("):")]
    depth = 0
    for i, c in enumerate(arg_str):
        if c == "," and depth == 0:
            return arg_str[i + 1:].strip()
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
    return ""

def move_import_inside_function(code):
    """
    move all import and import from to the inside of a function (to avoid problem when parsing args)
    """
    # Parse the code into an AST
    tree = ast.parse(code)

    # Find all top-level import statements
    imports = [node for node in tree.body if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]

    # Find the function definitions
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    # Remove top-level import statements
    tree.body = [node for node in tree.body if (not isinstance(node, ast.Import)) and (not isinstance(node, ast.ImportFrom))]

    # Move the import statements inside each function
    for function in functions:
        function.body = imports + function.body

    # Generate the modified code
    modified_code = ast.unparse(tree)
    return modified_code

def type_check(typ, obj):
    """
    Checks the object is the correct type. Supports only bool, int, float, str,
    and (possibly nested) lists of these

    From: https://github.com/microsoft/PythonProgrammingPuzzles/blob/v0.2/puzzle_generator.py
    """
    type_s = type_str(typ)  # convert to string if necessary

    nest_depth = type_s.count("List")
    assert (
        type_s.count("[") == nest_depth
    ), "type_check only supports List for now, no Sets, Dicts, Tuples, ..."

    assert type_s.startswith("List[" * nest_depth) and type_s.endswith("]" * nest_depth)
    base_type = {"bool": bool, "int": int, "float": float, "str": str}[
        type_s[5 * nest_depth : len(type_s) - nest_depth]
    ]

    def helper(depth, o):
        if depth == 0:
            return type(o) is base_type
        else:
            return type(o) is list and all(helper(depth - 1, i) for i in o)

    return helper(nest_depth, obj)


def type_str(ty: type) -> str:
    """
    Convert type ty to string.
    :param ty: str, typing.List[int] , typing.List[typing.List[bool]], etc.
    :return: string form of type, "str", "List[int]" , "List[List[bool]]", etc.

    From: https://github.com/microsoft/PythonProgrammingPuzzles/blob/v0.2/puzzle_generator.py
    """
    type_str = str(ty).replace("typing.", "")
    return type_str[8:-2] if type_str.startswith("<class '") else type_str




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

def scrap_f_g(list_pb):
    """
    scrap f and g from generated puzzles
    """

    list_f_g=[]
    for pb in list_pb:
        tree = ast.parse(pb)
        # Find all function definitions in the AST
        function_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        f = ast.unparse(function_defs[0])
        g = ast.unparse(function_defs[1])
        list_f_g.append([f,g])
    return list_f_g

def preprocessing_P3(split: str = "train", n_token_max: int =512, load_embedding = False,debug=False) -> list[dict]:
    """
    dl puzzles from P3 dataset and give train or test puzzles
    split = "train" or "test"
    """
    import sys 
    sys.set_int_max_str_digits(10_000)
    puzzles = requests.get(
        "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
    ).json()
    data_split = requests.get(
        "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/main/puzzles/split.json"
    ).json()
    enc = tiktoken.encoding_for_model("gpt-4")
    puzzles_set=[]
    generated_programs=[]
    for i in puzzles:
        if i["name"][:-2] in data_split[split] and i["sol_bodies"]!=[]:
            puzzle_2_add={}
            puzzle_2_add["f"] = add_return_bool_2_f(return_f(i))
            puzzle_2_add["g"] = return_g(i,puzzle_2_add["f"])
            puzzle_2_add['attempts'] = 1 # 
            puzzle_2_add["program_str"] = merge_Q_and_A([(puzzle_2_add["f"],puzzle_2_add["g"])])[0]
            generated_programs.append(puzzle_2_add["program_str"])
            
            
            results = pool_exec_processes(
                puzzle_2_add["program_str"],
                func_name="g",debug =True,
                processes=1
                )
            puzzle_2_add["result_obj"]=results[0]
            puzzles_set.append(puzzle_2_add)
    
    if split == "test":
        return puzzles_set
    else:
        List_len_embedding = []
        for puzz in puzzles_set:
            List_len_embedding.append(len(enc.encode(puzz["program_str"])))
        index=np.array(List_len_embedding)<=n_token_max
        #remove item where index is False
        puzzles_set = [item for i, item in enumerate(puzzles_set) if index[i]]
        if load_embedding:            
            import os
            script_dir = os.path.dirname(__file__) 
            path_embed = script_dir+"/preprocess_p3_emb.json"
            with open(path_embed, "r") as f:
                list_emb = json.load(f)
                list_program = [puzz["program_str"] for puzz in list_emb]
                # list_keys=list(list_emb.keys())
            for puzz in (puzzles_set):
                code = puzz["program_str"]
                if code in list_program:
                    idx = list_program.index(code)
                    emb = list_emb[idx]["emb"]
                    puzz["emb"] = emb
        if debug:
            for puzz in (puzzles_set):
                puzz["emb"] = np.random.randint(0, 2, 10)
        return puzzles_set
    
def get_limited_trainset():
    import os
    script_dir = os.path.dirname(__file__) 
    path_embed = script_dir+"/preprocess_p3_emb_3_puzzles.json"
    with open(path_embed, "r") as f:
        list_puzzle = json.load(f)
    return list_puzzle

    
def preprocessing_P3_no_test(split: str = "train", n_token_max: int =512, load_embedding = False,debug=False) -> list[dict]:
    """
    dl puzzles from P3 dataset and give train or test puzzles
    split = "train" or "test"
    """
    import sys 
    sys.set_int_max_str_digits(10_000)
    puzzles = requests.get(
        "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
    ).json()
    data_split = requests.get(
        "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/main/puzzles/split.json"
    ).json()
    enc = tiktoken.encoding_for_model("gpt-4")
    puzzles_set=[]
    generated_programs=[]
    for i in puzzles:
        if i["name"][:-2] in data_split[split] and i["sol_bodies"]!=[]:
            puzzle_2_add={}
            puzzle_2_add["f"] = add_return_bool_2_f(return_f(i))
            puzzle_2_add["g"] = return_g(i,puzzle_2_add["f"])
            puzzle_2_add['attempts'] = 1 # 
            puzzle_2_add["program_str"] = merge_Q_and_A([(puzzle_2_add["f"],puzzle_2_add["g"])])[0]
            puzzle_2_add["g_firstline"]= return_header_g(puzzle_2_add["f"])
            generated_programs.append(puzzle_2_add["program_str"])
            
            
            puzzles_set.append(puzzle_2_add)
    
    else:
        List_len_embedding = []
        for puzz in puzzles_set:
            List_len_embedding.append(len(enc.encode(puzz["program_str"])))
        index=np.array(List_len_embedding)<=n_token_max
        #remove item where index is False
        puzzles_set = [item for i, item in enumerate(puzzles_set) if index[i]]
        # if load_embedding:            
        #     import os
        #     script_dir = os.path.dirname(__file__) 
        #     path_embed = script_dir+"/preprocess_p3_emb.json"
        #     with open(path_embed, "r") as f:
        #         list_emb = json.load(f)
        #         list_program = [puzz["program_str"] for puzz in list_emb]
        #         # list_keys=list(list_emb.keys())
        #     for puzz in (puzzles_set):
        #         code = puzz["program_str"]
        #         if code in list_program:
        #             idx = list_program.index(code)
        #             emb = list_emb[idx]["emb"]
        #             puzz["emb"] = emb
        # if debug:
        #     for puzz in (puzzles_set):
        #         puzz["emb"] = np.random.randint(0, 2, 10)
        return puzzles_set
    
def sample_target_skill_smart(all_emb) -> list[bool]:
    """ 
    sample an skill to target for p3 problem
    all_emb: list of binary vector 
    target_skill: bool vector (same length as last dim of all_emb)
    """
    n_niches = np.shape(all_emb)[-1]
    all_emb_2_set= [tuple(emb) for emb in all_emb]
    all_emb_set = list(set(all_emb_2_set))
    binary_vectors = np.array(list(itertools.product([0, 1], repeat=n_niches)))#list of all possible niches
    out=cdist(binary_vectors, np.array(all_emb_set), metric='cityblock')
    density=(out==1).sum(axis=1) # find every niches within a distance of 1
    density=density*(out.min(axis=1)!=0) # remove already explored niches (sampling weight = 0)
    density_norm=density/np.sum(density)
    idx_niches_sampled=np.random.choice(len(binary_vectors),p=density_norm)
    target_skill=list(binary_vectors[idx_niches_sampled])
    return target_skill


def sample_fewshot_example(skill_targeted, all_emb, all_phenotypes, n_few_shot_example=3):
    """
    sample n_few_shot_example examples from closest example in the embedding space (from the archive)
    details: sample n_few_shot_example closest niches and sample one example with uniform probabilty from each niche  
    
    skill_targeted: bool vector
    all_emb: list of embedding associated to all_phenotypes
    all_phenotypes: list of P3 phenotypes object 
    n_few_shot_example: int, number of example to sample
    """
    list_few_shot_example_phenotypes= []
    list_coord_niches_sampled = []
    # choose puzzle from closest niches half from trainset
    all_emb_2_set= [tuple(emb) for emb in all_emb]
    all_emb_set = list(set(all_emb_2_set))
    dists = cdist([skill_targeted], all_emb_set)[0]
    # shuffle indices to have true uniform sampling of closest niches
    shuffled_indices = np.arange(len(dists))
    np.random.shuffle(shuffled_indices)
    nearest_niches = shuffled_indices[np.argsort(dists[shuffled_indices])]
    for idx in nearest_niches:
        emb_2_add = list(all_emb_set[idx])
        if not(emb_2_add in list_coord_niches_sampled):
            list_coord_niches_sampled.append(emb_2_add)
            list_2_sample = [i for i in all_phenotypes if emb_2_add == i.emb]
            idx_sample=np.random.choice(len(list_2_sample))
            list_few_shot_example_phenotypes.append(list_2_sample[idx_sample])
        if len(list_few_shot_example_phenotypes)>=n_few_shot_example:
            break
    return list_few_shot_example_phenotypes