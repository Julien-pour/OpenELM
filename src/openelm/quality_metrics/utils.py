from typing import List
import ast
import re
import numpy as np
from tqdm import tqdm

import torch.multiprocessing as mp
import torch
from transformers import CodeLlamaTokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import os
# dedup,info,silence_std_err
from copy import copy, deepcopy


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

def return_header_g(f):
    args_f = extract_args_f(f)
    return "def g(" + args_f + "):"


def return_g(puzzle_json, f):
    if not puzzle_json["sol_bodies"]:
        print("no solution in json")
        return "def g(""):\n    pass"
    args_f = extract_args_f(f)
    g = "def g(" + args_f + "):\n" + puzzle_json["sol_bodies"][0]
    return g

def merge_Q_and_A(liste_fg):
    parsed = deepcopy(liste_fg) # format [(f,g),(f,g),...]

    judge_srcs = [f"{f}\n{g}\nassert f(g())" for (f, g) in parsed] # format the code to be judged
    return judge_srcs

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





def info(*args, **kwargs):
    _get_or_create_logger().info(print_to_string(*args, **kwargs))

def dedup(stuff):
    seen = set()
    return [a for a in stuff if a not in seen and not seen.add(a)]

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

### general utils
def load_prompt_PP(one_shot_prompt_id):
    utils_directory = os.path.dirname(os.path.realpath(__file__))
    # path_prompt = os.path.abspath("src/examplefile.txt") which one is "better"?
    path_prompt = os.path.join(utils_directory,'dataset_progress', one_shot_prompt_id)
    with open(path_prompt, 'r') as f:
        prompt_text = f.read()
    return prompt_text

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


### puzzle utils


def make_solution(puzzle):
    # chooses one solution among the available ones
    header = puzzle['sol_header'].replace('def sol(', 'def g(')
    # body = np.random.choice(puzzle['sol_bodies'])  # choose at random
    body = puzzle['sol_bodies'][0]  # choose the first one, ideal to get
    return '\n'.join([header, body])


def make_puzzle(puzzle, include_docstring=False):
    if include_docstring:
        splitlines = puzzle['sat'].split('\n')
        splitlines.insert(1, puzzle['sol_docstring'])
        puz_str = '\n'.join(splitlines)
    else:
        puz_str = puzzle['sat']
    return puz_str.replace('def sat(', 'def f(')


def parse_puzzle_from_str(s):
    try:
        functions = [el for el in ast.parse(s).body if isinstance(el, ast.FunctionDef)]
        f = ast.unparse(functions[0])
        g = ast.unparse(functions[1])
        return f, g
    except:
        return '', ''

### transformer utils


def create_model_and_tokenizer(model_id, compile=True, dtype=torch.bfloat16, flash_attn=True):
    if 'codellama' in model_id:
        tokenizer = CodeLlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    elif 'llama' in model_id:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # todo: simplify
    if flash_attn:
        try:
            import flash_attn
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                # quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                # quantization_config=quantization_config,
                device_map="auto",
                # local_files_only=True,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            # quantization_config=quantization_config,
            device_map="auto",
            # local_files_only=True,
            trust_remote_code=True
        )
    # model.cuda()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    if compile:
        model = torch.compile(model)

    return model, tokenizer


def remove_unnecessary_indices(tokenized_text):
    attention_unmask = 1 - tokenized_text.attention_mask
    first_index = attention_unmask.sum(-1).min()
    attention_mask = tokenized_text.attention_mask[:, first_index:]
    input_ids = tokenized_text.input_ids[:, first_index:]
    tokenized_text.input_ids = input_ids
    tokenized_text.attention_mask = attention_mask
    return tokenized_text


def get_solution_mask(full_prompt, solution, return_list=False):
    # given an iterable of indices corresponding to the full prompt with the solution and one corresponding
    # to the solution tokens, return the attention mask for the solution
    # find the start and end idx of the longest overlapping sequence in solution

    def number_overlapping(seq1, seq2):
        # must be contiguous
        num = 0
        start_idx = 0
        for i in range(1, min(len(seq1), len(seq2))):
            s1, s2 = seq1[i], seq2[i]
            if s1 == s2:
                num += 1
            else:
                num = 0
                start_idx = i
        return num, start_idx

    max_num_overlapping = 0
    best_start_idx = 0
    for start_idx in range(len(full_prompt)):
        num_overlapping, s_idx = number_overlapping(full_prompt[start_idx:], solution)
        if num_overlapping >= max_num_overlapping:
            max_num_overlapping = num_overlapping
            best_start_idx = start_idx + s_idx

    attention_list = [0 for _ in range(len(full_prompt))]
    for idx in range(best_start_idx, best_start_idx + max_num_overlapping + 1):
        attention_list[idx] = 1

    if isinstance(full_prompt, torch.Tensor) and not return_list:
        return torch.Tensor(attention_list).to(full_prompt.device)
    else:
        return attention_list
    # cast to the right type


def get_solution_mask_loop(args):
    full_prompts, solutions = args
    results = []
    for full_prompt, sol in zip(full_prompts, solutions):
        results.append(get_solution_mask(full_prompt, sol, return_list=True))

    return results


def split_samples(samples, num_workers):
    num_samples = len(samples)
    divisor = num_samples // num_workers
    remainder = num_samples % num_workers

    split = []
    for rank in range(num_workers):
        if rank < remainder:
            start_idx = rank * (divisor + 1)
            end_idx = start_idx + divisor + 1
        else:
            over_remainder = rank - remainder
            start_idx = remainder * (divisor + 1) + over_remainder * divisor
            end_idx = start_idx + divisor
        split.append(samples[start_idx:end_idx])

    return split


def get_all_solution_masks(archive_tokenized_puzzles, solutions_tokenized, num_workers=None):
    solution_attention_mask = torch.zeros_like(archive_tokenized_puzzles.attention_mask)
    # compute the solution attention mask

    print('Getting attention masks:')
    if num_workers is None:
        for idx, (full_prompt, sol) in tqdm(enumerate(zip(archive_tokenized_puzzles.input_ids,
                                                          solutions_tokenized.input_ids))):
            mask = get_solution_mask(full_prompt, sol)
            solution_attention_mask[idx] = mask

    else:
        # divide the tokenized data
        # todo check there is no issue with the masks
        archive_tokenized_puzzles_split = split_samples(archive_tokenized_puzzles.input_ids, num_workers)
        solutions_tokenized_split = split_samples(solutions_tokenized.input_ids, num_workers)
        args = list(zip(archive_tokenized_puzzles_split, solutions_tokenized_split))
        processes = []

        # might be better with a queue
        with mp.Pool(num_workers) as p:
            results = p.map(get_solution_mask_loop, args)
            print("Map finished")

        i = 0
        for el in results:
            solution_attention_mask[i:i+len((el))] = torch.Tensor(el)
            i += len(el)

        return solution_attention_mask


def get_solution_mask_from_str(full_prompt: str, solution: str, tokenizer, num_solution_tokens: int,
                               num_total_tokens, return_type='pt'):
    # should be parallelizable (saves in the tokenizer)
    assert solution in full_prompt
    # use pattern matching to get the text before the solution
    pattern = f'(.*){solution}(.*)'
    match = re.match(pattern, full_prompt)
    assert match is not None  # should never happen
    # count tokens
    num_tokens_before = len(tokenizer(match[0]))
    # create mask
    if return_type == 'pt':
        mask = torch.zeros(num_total_tokens)
        mask[num_tokens_before:num_tokens_before+num_solution_tokens] = 1.
    else:
        mask = [0] * len(num_total_tokens)
        for i in range(num_solution_tokens):
            mask[num_tokens_before+i] = 1
    return mask


def get_solution_mask_from_str_loop(full_prompts, solutions, tokenizer, num_solution_tokenss,
                                    archive_attention_mask, offsets):
    # offset is due to padding (there might be a way to bypass using it)
    matches = [full_prompt.split(solution)[0] for solution, full_prompt in zip(solutions, full_prompts)]
    num_tokens_before = [len(t) for t in tokenizer(matches).input_ids]
    masks = torch.zeros_like(archive_attention_mask)
    for i, (t, num_solution_tokens, o) in enumerate(zip(num_tokens_before, num_solution_tokenss, offsets)):
        masks[i, o+t:o+t+num_solution_tokens] = 1.

    return masks


REF_PUZZLE = '''def sat(s: List[str]):
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)'''

REF_PUZZLE_NODOC = '''def sat(s: List[str]):
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)'''

REF_SOL = '''def sol():
    return ["a" * (i + 2) + "b" for i in range(1000)]'''

HANOI_PUZZLE = '''def sat(moves: List[List[int]]):
    """
    Eight disks of sizes 1-8 are stacked on three towers, with each tower having disks in order of largest to
    smallest. Move [i, j] corresponds to taking the smallest disk off tower i and putting it on tower j, and it
    is legal as long as the towers remain in sorted order. Find a sequence of moves that moves all the disks
    from the first to last towers.
    """
    rods = ([8, 7, 6, 5, 4, 3, 2, 1], [], [])
    for [i, j] in moves:
        rods[j].append(rods[i].pop())
        assert rods[j][-1] == min(rods[j]), "larger disk on top of smaller disk"
    return rods[0] == rods[1] == []'''

HANOI_PUZZLE_NODOC = '''def sat(moves: List[List[int]]):
    """
    Eight disks of sizes 1-8 are stacked on three towers, with each tower having disks in order of largest to
    smallest. Move [i, j] corresponds to taking the smallest disk off tower i and putting it on tower j, and it
    is legal as long as the towers remain in sorted order. Find a sequence of moves that moves all the disks
    from the first to last towers.
    """
    rods = ([8, 7, 6, 5, 4, 3, 2, 1], [], [])
    for [i, j] in moves:
        rods[j].append(rods[i].pop())
        assert rods[j][-1] == min(rods[j]), "larger disk on top of smaller disk"
    return rods[0] == rods[1] == []'''

HANOI_SOL = '''def sol():
    moves = []
    def hanoi(n, source, temp, dest):
        if n > 0:
            hanoi(n - 1, source, dest, temp)
            moves.append([source, dest])
            hanoi(n - 1, temp, source, dest)
    hanoi(8, 0, 1, 2)
    return moves'''


# embedding utils


def embed_puzzle(tokenizer, model, p):
    with torch.no_grad():
        tokens = tokenizer(p['sat'], return_tensors='pt')
        if tokens.input_ids.shape[1] > 2048:
            return None
        emb = model(
            input_ids=tokens.input_ids.to('cuda'),
            attention_mask=tokens.attention_mask.to('cuda'),
            output_hidden_states=True,
        ).hidden_states[-1][:, -1].cpu().tolist()
    return emb


@torch.no_grad()
def embed_puzzles(tokenizer, model, texts, batch_size, out_type='tensor'):
    device = model.device
    hidden_size = model.config.hidden_size
    if out_type == 'tensor':
        embeddings = torch.zeros(len(texts), hidden_size)
    else:
        embeddings = []
    tokens = tokenizer(texts, return_tensors='pt', padding=True)
    for index in tqdm(range(0, len(tokens.input_ids), batch_size)):
        input_ids = tokens.input_ids[index:index+batch_size]
        attention_mask = tokens.attention_mask[index:index+batch_size]
        embs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
        ).hidden_states[-1].mean(1).cpu()

        if out_type == 'tensor':
            embeddings[index:index+batch_size] = embs
        else:
            embeddings += embs

    return embeddings


cos = torch.nn.CosineSimilarity(dim=-1)


def dotprod(a, b):
    return (a * b).sum(-1)


def cosine_similarity_matrix(a, b, eps=1e-8):  # untested
    assert len(a.shape) == len(b.shape) == 2
    a = a.unsqueeze(1)
    b = b.unsqueeze(0)
    norma = a.pow(2).sum(-1).pow(0.5)
    normb = b.pow(2).sum(-1).pow(0.5)
    norm_mat = norma * normb
    norm_mat = torch.maximum(norm_mat, torch.ones_like(norm_mat) * eps)
    prod = dotprod(a, b)
    c = prod / norm_mat
    return c


def pairwise_distance(a, b):
    assert len(a.shape) == len(b.shape) == 2
    a = a.unsqueeze(1)
    b = b.unsqueeze(0)
    distance = (a - b).pow(2).sum(-1).pow(0.5)
    return distance




### utils for logging
import logging
import io
import sys

_configured = False
my_path = os.path.dirname(__file__)


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
