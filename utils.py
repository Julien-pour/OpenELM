import os
import io
import gc
import sys
import ast
import json
import numpy as np
import pickle
from tqdm import tqdm
from copy import copy, deepcopy
import torch

import logging
import inspect

my_path = os.path.dirname(__file__)


def getallitems(maps):
    """
    Returns all the phenotypes that are in the Map."""
    genomes = maps["genomes"]
    valid_phenotype=[]
    for gen in np.ndindex(genomes.shape):
        value_gen = type(genomes[gen])
        if value_gen!=float and value_gen!=int:
            valid_phenotype.append(genomes[gen])

    return valid_phenotype

# init maps
def return_cells_filled_per_gen_map_elite(path_save_all,max_gen=-1,include_trainset=False,include_full_trainset=False):
    #_init_discretization():
    n_skills=10
    behavior_space= np.repeat([[0, 1]], n_skills, axis=0).T

    bins = np.linspace(*behavior_space,  3)[1:-1].T  # type: ignore
    # bins
    def to_mapindex(b, bins=bins) :
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, bins))
        )
        

    path_maps = path_save_all.split("save_all.json")[0]+"maps.pkl"

    with open(path_maps, "rb") as f:
        maps = pickle.load(f)
    allitems=getallitems(maps)
    items_trainset = [item for item in allitems if item.idx_generation==-1]
    items_gen = [item for item in allitems if item.idx_generation!=-1]
    if "/elm/" in path_save_all and "run_saved" in path_save_all:
        elm_phenotype = path_save_all.split("maps.pkl")[0]+"_phenotype.npy"
        phen= np.load(elm_phenotype)
        for i in range(len(items_gen)):
            items_gen[i].emb=phen[i]
    print(len(items_gen))
    
    nonzero=np.zeros(shape=[2]*n_skills,dtype=bool)#np.zeros_like(maps["nonzero"]) 
    # list_map_ix_train=[to_mapindex(puzz.emb) for puzz in items_trainset]
    if include_full_trainset:
        path_trainset = "/media/data/flowers/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
        with open(path_trainset, "r") as f:
            list_puzzle_full_trainset = json.load(f)
            list_map_ix_train=[to_mapindex(puzz["emb"]) for puzz in list_puzzle_full_trainset]
            for map_ix_train in list_map_ix_train:
                nonzero[map_ix_train] = True
    # elif include_trainset:
    #     for map_ix_train in list_map_ix_train:
    #         nonzero[map_ix_train] = True

    # separate items per generation
    list_gens= [puzz.idx_generation for puzz in items_gen]
    if max_gen==-1:
        max_gen = max(list_gens)
    
    list_emb_per_idx_gen = [[] for _ in range(max_gen+1)]
    for i, gen in enumerate(list_gens):
        if gen<len(list_emb_per_idx_gen):
            list_emb_per_idx_gen[gen].append(items_gen[i].emb)
        
    number_of_cells_filled=[nonzero.sum()]
    for i in range(len(list_emb_per_idx_gen)):
        for puzz_emb in list_emb_per_idx_gen[i]:
            new_map_ix=to_mapindex(puzz_emb)
            nonzero[new_map_ix] = True
        number_of_cells_filled.append(nonzero.sum())
    return number_of_cells_filled


def return_cells_filled_per_puz_gen_map_elite(path_save_all,max_gen=-1,include_trainset=False,include_full_trainset=False):
    #_init_discretization():
    n_skills=10
    behavior_space= np.repeat([[0, 1]], n_skills, axis=0).T

    bins = np.linspace(*behavior_space,  3)[1:-1].T  # type: ignore
    # bins
    def to_mapindex(b, bins=bins) :
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, bins))
        )
        

    path_maps = path_save_all.split("save_all.json")[0]+"maps.pkl"

    with open(path_maps, "rb") as f:
        maps = pickle.load(f)
    allitems = getallitems(maps)
    items_trainset = [item for item in allitems if item.idx_generation==-1]
    items_gen = [item for item in allitems if item.idx_generation!=-1]
    if "/elm/" in path_save_all and "run_saved" in path_save_all:
        elm_phenotype = path_save_all.split("maps.pkl")[0]+"_phenotype.npy"
        phen= np.load(elm_phenotype)
        for i in range(len(items_gen)):
            items_gen[i].emb=phen[i]
    print(len(items_gen))
    
    nonzero=np.zeros(shape=[2]*n_skills,dtype=bool)  # np.zeros_like(maps["nonzero"])
    # list_map_ix_train=[to_mapindex(puzz.emb) for puzz in items_trainset]
    if include_full_trainset:
        path_trainset = "/media/data/flowers/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
        with open(path_trainset, "r") as f:
            list_puzzle_full_trainset = json.load(f)
            list_map_ix_train=[to_mapindex(puzz["emb"]) for puzz in list_puzzle_full_trainset]
            for map_ix_train in list_map_ix_train:
                nonzero[map_ix_train] = True
    # elif include_trainset:
    #     for map_ix_train in list_map_ix_train:
    #         nonzero[map_ix_train] = True

    # separate items per generation
    list_gens = [puzz.idx_generation for puzz in items_gen]
    if max_gen == -1:
        max_gen = max(list_gens)
    
    list_emb_per_idx_gen = [[] for _ in range(max_gen+1)]
    for i, gen in enumerate(list_gens):
        if gen < len(list_emb_per_idx_gen):
            list_emb_per_idx_gen[gen].append(items_gen[i].emb)
        
    number_of_cells_filled=[nonzero.sum()]
    for i in range(len(list_emb_per_idx_gen)):
        for puzz_emb in list_emb_per_idx_gen[i]:
            new_map_ix=to_mapindex(puzz_emb)
            nonzero[new_map_ix] = True
            number_of_cells_filled.append(nonzero.sum())
    return number_of_cells_filled



def return_cells_filled_in_embspace_from_NLPembspace(path_save_all,path_centroids,centroids=None,max_gen=-1,include_trainset=False,include_full_trainset=False,model=None,tokenizer=None,pipeline=None):
    #_init_discretization():
    if centroids is None:
        centroids=np.load(path_centroids)
    # bins
    def to_mapindex(b,centroids=centroids):
        """Maps a phenotype (position in behaviour space) to the index of the closest centroid."""
        return (
            None
            if b is None
            else (np.argmin(np.linalg.norm(b - centroids, axis=1)),)
        )
        
    path_maps_cvt = path_centroids.split("centroids.npy")[0]+"maps.pkl"
    path_maps = path_save_all.split("save_all.json")[0]+"maps.pkl"

    with open(path_maps_cvt, "rb") as f:
        maps_cvt = pickle.load(f)
    with open(path_maps, "rb") as f:
        maps = pickle.load(f)
    allitems=getallitems(maps)
    with torch.no_grad():
        for i in tqdm(range(len(allitems))):
            program_str=allitems[i].program_str
            if pipeline is None:
                inputs = tokenizer.encode(program_str, return_tensors="pt",truncation=True,max_length=512)
                emb = model(inputs.to("cuda"))[0]
                allitems[i].emb=emb.to("cpu").numpy()
            else:
                features = np.array(pipeline(program_str))
                allitems[i].emb=features.mean(axis=1).flatten()
    items_trainset = [item for item in allitems if item.idx_generation==-1]
    items_gen = [item for item in allitems if item.idx_generation!=-1]

    print(len(items_gen))

    nonzero=np.zeros_like(maps_cvt["nonzero"]) 
    list_map_ix_train=[to_mapindex(puzz.emb) for puzz in items_trainset]
    
    if include_full_trainset:
        path_trainset = "/media/data/flowers/OpenELM/src/openelm/utils/preprocess_p3_emb.json"
        with open(path_trainset, "r") as f:
            list_puzzle_full_trainset = json.load(f)
            # list_puzzle_full_trainset["program_str"]
        with torch.no_grad():
            for i in tqdm(range(len(list_puzzle_full_trainset))):
                program_str=list_puzzle_full_trainset[i]["program_str"]
                if pipeline is None:
                    inputs = tokenizer.encode(program_str, return_tensors="pt",truncation=True,max_length=512)
                    emb = model(inputs.to("cuda"))[0]
                    emb=emb.to("cpu").numpy()
                else:
                    features = np.array(pipeline(program_str))
                    emb=features.mean(axis=1).flatten()
                list_puzzle_full_trainset[i]["emb"]=emb
        list_map_ix_train=[to_mapindex(puzz["emb"]) for puzz in list_puzzle_full_trainset]
        for map_ix_train in list_map_ix_train:
            nonzero[map_ix_train] = True
              
    elif include_trainset:
        for map_ix_train in list_map_ix_train:
            nonzero[map_ix_train] = True

    # separate items per generation
    list_gens = [puzz.idx_generation for puzz in items_gen]
    if max_gen == -1:
        max_gen = max(list_gens)
    
    list_emb_per_idx_gen = [[] for _ in range(max_gen+1)]
    for i, gen in enumerate(list_gens):
        if gen < len(list_emb_per_idx_gen):
            list_emb_per_idx_gen[gen].append(items_gen[i].emb)
        
    number_of_cells_filled=[nonzero.sum()]
    for i in range(len(list_emb_per_idx_gen)):
        for puzz_emb in list_emb_per_idx_gen[i]:
            new_map_ix = to_mapindex(puzz_emb)
            nonzero[new_map_ix] = True
        number_of_cells_filled.append(nonzero.sum())
    return number_of_cells_filled


### utils for eval


def test_puzzle(f, x):
    """Checks if x is of the correct type and makes f return True (literally True, not an integer or whatever)

    :param f: Puzzle
    :param x: candidate answer
    :return:
    """
    answer_type = list(f.__annotations__.values())[0]
    if not type_check(x, answer_type):
        raise TypeError
    return f(x) is True


def type_check(obj, typ):
    """
    check if obj is of type `typ` where `typ` is a `typing` module type annotation, eg List[int]
    The way we do this to be compatible across versions is we first convert the type to a string.
    """

    type_str = str(typ).replace("typing.", "")
    if type_str.startswith("<class '"):
        type_str = type_str[8:-2]

    def helper(obj, type_st: str):
        """test if obj is of type type_st"""
        t = {"str": str, "int": int, "float": float, "bool": bool}.get(type_st)
        if t is not None:
            return type(obj) == t
        assert type_st.endswith("]"), f"Strange type `{type_st}`"
        inside = type_st[type_st.index("[")+1:-1].split(", ")
        if type_st.startswith("List["):
            [i] = inside
            return isinstance(obj, list) and all(type_check(elem, i) for elem in obj)
        if type_st.startswith("Set"):
            [i] = inside
            return isinstance(obj, set) and all(type_check(elem, i) for elem in obj)
        print(f"type not handled: {typ}")
        return True

    return helper(obj, type_str)


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


def dedup(stuff):
    seen = set()
    return [a for a in stuff if a not in seen and not seen.add(a)]


def color_str(obj, code="\033[0;36m"):
    return code + str(obj) + '\033[0m'



### utils for logging


_configured = False


def configure_logging(stdio_level=logging.INFO,
                      file_level=logging.DEBUG,
                      path=os.path.join(my_path, "../logs/"),
                      filename=os.path.basename(sys.argv[0]).replace(".py", "") + ".log"):
    global _configured
    if _configured:
        warning("Re-configuring logging")
    # create path if necessary
    os.makedirs(path, exist_ok=True)
    stdio_handler = logging.StreamHandler()
    stdio_handler.setLevel(stdio_level)
    file_hanlder = logging.FileHandler(os.path.join(path, filename))
    file_hanlder.setLevel(file_level)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(name)s:%(message).512s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=min(stdio_level, file_level),
        handlers=[stdio_handler, file_hanlder]
    )

    _configured = True
    _get_or_create_logger().debug("Configured logging")


_loggers = {}


def _get_or_create_logger():
    global _configured, _loggers
    if not _configured:
        configure_logging()
    name = "_"
    for frame in inspect.stack():
        name = inspect.getmodule(frame[0]).__name__
        if name != __name__:
            break
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


_std_errs = None


def silence_std_err(quiet=True):
    global _std_errs
    if _std_errs is None:
        _std_errs = {"orig": os.dup(2), "devnull": os.open(os.devnull, os.O_RDWR)}
    if quiet:
        os.dup2(_std_errs["devnull"], 2)  # to avoid printing the s_push parser when parsing stuff with "((((()))))"
    else:
        os.dup2(_std_errs["orig"], 2)


def print_to_string(*args, end="", **kwargs):
    with io.StringIO() as buf:
        print(*args, file=buf, end=end, **kwargs)
        return buf.getvalue()


def debug(*args, **kwargs):
    _get_or_create_logger().debug(print_to_string(*args, **kwargs))


def info(*args, **kwargs):
    _get_or_create_logger().info(print_to_string(*args, **kwargs))


log = info


def warning(*args, **kwargs):
    _get_or_create_logger().warning(print_to_string(*args, **kwargs))


warn = warning


def error(*args, **kwargs):
    _get_or_create_logger().error(print_to_string(*args, **kwargs))


def merge_Q_and_A(liste_fg):
    parsed = deepcopy(liste_fg) # format [(f,g),(f,g),...]

    judge_srcs = [f"{f}\n{g}\nassert f(g())" for (f, g) in parsed] # format the code to be judged
    return judge_srcs


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


def return_f(puzzle_json):
    puzzle_json = deepcopy(puzzle_json)
    f = puzzle_json["sat"]
    #  add 'sol_docstring' (description of the problem) to the function f
    f = f.replace("sat(", "f(")
    idx_add_problem_description = f.find("\n")

    if type(puzzle_json["sol_docstring"]) == str:
        f=f[:idx_add_problem_description+1]+ puzzle_json["sol_docstring"]+"\n"+f[idx_add_problem_description+1:]
    return f


def add_return_bool_2_f(f):
    tree = ast.parse(f)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.returns = ast.Name(id='bool', ctx=ast.Load())

    return ast.unparse(tree)


def return_header_g(f):
    args_f = extract_args_f(f)
    return "def g(" + args_f + "):"


def return_g(puzzle_json, f):
    if not puzzle_json["sol_bodies"]:
        print("no solution in json")
        return "def g(""):\n    pass"
    args_f = extract_args_f(f)
    g = "def g(" + args_f + "):\n" + deepcopy(puzzle_json["sol_bodies"])[0]
    return g


def preprocessing_p3(puzzles, n_token_max: int = 512, path=None, tokenizer=None) -> list[dict]:
    """
    dl puzzles from P3 dataset and give train or test puzzles
    split = "train" or "test"
    """
    puzzles_set = []
    generated_programs = []
    for i in puzzles:
        puzzle_2_add = {}
        puzzle_2_add["f"] = add_return_bool_2_f(return_f(i))
        puzzle_2_add["g"] = return_g(i, puzzle_2_add["f"])
        puzzle_2_add['attempts'] = 0
        puzzle_2_add["program_str"] = merge_Q_and_A([(puzzle_2_add["f"],puzzle_2_add["g"])])[0]
        puzzle_2_add["g_firstline"] = return_header_g(puzzle_2_add["f"])
        generated_programs.append(puzzle_2_add["program_str"])
        puzzles_set.append(puzzle_2_add)

    list_len_embedding = []
    for puzz in puzzles_set:
        len_puzz = len(tokenizer(puzz["program_str"], return_tensors="pt")["input_ids"][0])
        # print(len_puzz)
        list_len_embedding.append(len_puzz)
    index = np.array(list_len_embedding)<=n_token_max
    # remove item where index is False
    puzzles_set = [item for i, item in enumerate(puzzles_set) if index[i]]
    print("puzzle found =", len(puzzles_set))
    return puzzles_set


def clean_cache():  # use this after del ing model and tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.cpu()
    gc.collect()
    torch.cuda.empty_cache()


from concurrent.futures import ThreadPoolExecutor
def get_completion(client, prompt : str, cfg_generation,temperature=None)->str:
    """Get completion from OpenAI API"""
    kwargs={}
    kwargs.update(cfg_generation)
    if temperature is not None:
        kwargs["temperature"]= temperature
    try :
        completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI programming assistant"},#You are a coding assistant, skilled in writting code with creative flair."},
            {"role": "user", "content": prompt}
        ],**kwargs
        )
    except Exception as e:
        print("completion problem: ",e)
        return None 
    # completion_token = completion.usage.completion_tokens
    # prompt_token = completion.usage.prompt_tokens
            
    out = completion.choices[0].message.content
    return out




def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_multiple_completions(client, batch_prompt: list[str], cfg_generation: dict, batch_tools: list[list[dict]]=None,max_workers=20,temperature=None)->list[str]:
    """Get multiple completions from OpenAI API
    batch_tools =[[tools]] tools is the function, toll_name is the name of the tool

                    /!\ need to integrate batch tools in the loop /!\
    """
    # check that batch_prompt is list[str]
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    completions = []
    if max_workers>1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch in chunks(batch_prompt, max_workers):
                for idx,message_list in enumerate(sub_batch):
                    # kwargs_modified = args.copy()
                    # kwargs_modified["messages"] = message_list
                    kwargs = {"client":client, "prompt":message_list}
                    kwargs["cfg_generation"]=cfg_generation
                    if temperature is not None:
                        kwargs["temperature"]= temperature
                    # if "kwargs" in kwargs_modified:
                    #     original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(
                        get_completion,**kwargs
                    )
                    completions.append(future)
        # Retrieve the results from the futures
        results = [future.result() for future in completions]
    else:
        for idx,message_list in enumerate(batch_prompt):
            # kwargs_modified = args.copy()
            # kwargs_modified["messages"] = message_list
            kwargs = {"client":client, "prompt":message_list}
            kwargs["cfg_generation"]=cfg_generation
            if temperature is not None:
                kwargs["temperature"]= temperature
            # if "kwargs" in kwargs_modified:
            #     original_kwargs = kwargs_modified.pop("kwargs")
            result = get_completion(**kwargs)
            completions.append(result)
            results = completions

    return results
