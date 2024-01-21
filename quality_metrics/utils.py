from typing import List
import ast
import re
import numpy as np
from tqdm import tqdm

import torch.multiprocessing as mp
import torch
from transformers import CodeLlamaTokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


### general utils


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


bi = "zou"

### transformer utils


def create_model_and_tokenizer(model_id, compile=True, dtype=torch.bfloat16):
    if 'codellama' in model_id:
        tokenizer = CodeLlamaTokenizer.from_pretrained(model_id, local_files_only=True)
    elif 'llama' in model_id:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        # quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
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

