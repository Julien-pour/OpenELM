from typing import List
import ast
import numpy as np
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


def get_solution_mask(full_prompt, solution):
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

    if isinstance(full_prompt, torch.Tensor):
        return torch.Tensor(attention_list).to(full_prompt.device)
    else:
        return attention_list
    # cast to the right type



REF_PUZZLE = '''def sat(s: List[str]):
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)'''

REF_PUZZLE_NODOC = '''def sat(s: List[str]):
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)'''

REF_SOL = '''def sol():
    return ["a" * (i + 2) + "b" for i in range(1000)]'''
