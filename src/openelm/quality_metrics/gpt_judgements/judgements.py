import os
import re
import json
from typing import List
from pprint import pprint

import string
import random

import numpy as np
from openai import OpenAI

from openelm.mutation_model import get_completion, get_multiple_completions

from openelm.quality_metrics.utils import make_puzzle, make_solution
from openelm.quality_metrics.gpt_judgements.prompts import (
    five_fold_ranking_prompt,
    random_permutation_prompt,
    five_fold_ranking_prompt_scrambled_cot
) 


TRAIN_PUZZLES = json.load(open('puzzles_train.json', 'r'))
PREFERENCE_PUZZLES = json.load(open('puzzles_with_prefs.json', 'r'))


example_template = """Puzzle and solution {i}:
```python
{puzzle}

{solution}
```"""

example_template_docstring_before = """Puzzle and solution {i}:
```python
Problem description: {docstring}

{puzzle}

{solution}
```"""


random_puzzle_labels = [
    'fjnw', 'pfmr', 'qpfc', 'wphv', 'kqsr'
]


def build_prompt(puzzle_strs: List[str]):
    puzzle_text = ''
    for i, p in enumerate(puzzle_strs): 
        puzzle_text += 'Puzzle {i}:\n' + p + '\n'
    return five_fold_ranking_prompt.format(examples=puzzle_text)


def get_example_string(doctstring_mode='in_puzzle', mode='normal', puzzle_set='first_five'):
    # docstring in {'in_puzzle', 'before_puzzle', 'none'} 
    ex_string = ''
    examples = []
    sampled_puzzles = []
    match puzzle_set:
        case 'first_five':
            for i in range(5):
                sampled_puzzles.append(TRAIN_PUZZLES[i])
            ids = list(range(5))
        case 'preferences':
            ids = np.random.choice(
                list(range(len(PREFERENCE_PUZZLES))),
                5,
                replace=False
            ).tolist()
            sampled_puzzles = [PREFERENCE_PUZZLES[i] for i in ids]
    
    # record permutation
    sampled_puzzles = list(enumerate(sampled_puzzles))
    sampled_puzzles = np.random.permutation(sampled_puzzles).tolist()
    permutation = [i for i, _ in sampled_puzzles]
    sampled_puzzles = [p for _, p in sampled_puzzles]

    if mode == 'scrambled':
        str_ids = []
        for _ in range(5):
            # generate a random seq of letters: 
            str_id = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
            str_ids.append(str_id)
    else:
        str_ids = None

    for i, p in enumerate(sampled_puzzles):

        if mode == 'normal':
            I = i
        else: 
            I = str_ids[i]

        match doctstring_mode:
            case 'in_puzzle':
                puzzle = make_puzzle(p, include_docstring=True)
                solution = make_solution(p)
                examples.append(example_template.format(i=I, puzzle=puzzle, solution=solution))
            case 'before_puzzle':
                puzzle = make_puzzle(p, include_docstring=False)
                solution = make_solution(p)
                doctstring = p['sol_docstring']
                examples.append(example_template_docstring_before.format(
                    i=I, puzzle=puzzle, solution=solution, docstring=doctstring.strip()))
            case _:
                puzzle = make_puzzle(p, include_docstring=False)
                solution = make_solution(p)
                examples.append(example_template.format(i=I, puzzle=puzzle, solution=solution))

    return '\n\n'.join(examples), permutation, str_ids, ids


def get_five_fold_results(client, cfg_generation, docstring_mode='in_puzzle', mode='normal',
                          puzzle_set='first_five'):

    match mode:
        case 'normal':
            prompt_template = five_fold_ranking_prompt
        case 'scrambled':
            prompt_template = five_fold_ranking_prompt_scrambled_cot

    # sample puzzles
    example_str, permutation, str_ids, ids = get_example_string(docstring_mode, mode=mode, 
                                                                puzzle_set=puzzle_set)
    prompt = prompt_template.format(examples=example_str)
    completion = get_completion(client, prompt, cfg_generation)
    return completion, prompt, permutation, str_ids, ids


def get_default_config():
    key = os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=key, max_retries=5, timeout=50)

    cfg_generation={"temperature": 0.0,
            "top_p": 1,
            # TODO: rename config option?
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
    }
    
    return client, cfg_generation


def test_validity(N=5, mode='normal', puzzle_set='first_five'):
    # Tests to see whether the model produces parsable lists, and consistent outputs
    client, cfg_generation = get_default_config()
    list_outputs = []
    justifications = []
    permutations = []
    prompts = []

    if puzzle_set == 'preferences':  # to recover which puzzle we sampled
        puz_ids = []

    for i in range(N):
        print(f'Completion {i}:')
        completion, prompt, permutation, str_ids, ids = get_five_fold_results(
            client, cfg_generation, docstring_mode='before_puzzle', mode=mode, 
            puzzle_set=puzzle_set)
        m = re.match('(.*)(\[.*\]).*', completion, re.DOTALL)
        if m is None:
            print('No match for list')
            print(f'Completion is: {completion}')
            continue
        else:
            string_list = m[2]
            comment = m[1]

        try:
            result = eval(string_list)
        except Exception as e:
            print("Evaluation failed with exception:")
            print(e)
            continue

        if not isinstance(result, list):
            print(f"The result doesn't have the required type (it's a {type(result)})")
        else:

            if mode == 'scrambled':
                # try to parse the results back into a list of indices.
                res = []
                try:
                    for el in result:
                        idx = str_ids.index(el)
                        res.append(idx)
                except Exception as e:
                    print(f'Couldn\'t find "{el} in {str_ids}"')
                    print(e)
                    continue
                result = res

            list_outputs.append((i, result))
            if comment:
                justifications.append((i, comment))
            permutations.append((i, permutation))
            prompts.append(prompt)

            if puzzle_set == 'preferences':
                puz_ids.append(ids)
                

    unique = []
    for _, el in list_outputs:
        if el not in unique:
            unique.append(el)
    print(f'Number of different elements: {len(unique)}')
    pprint(list_outputs)

    data = dict(unique=unique, list_outputs=list_outputs, justifications=justifications, 
                permutations=permutations, prompts=prompts)
    
    if puzzle_set == 'preferences':
        data['puz_ids'] = puz_ids

    save_name = 'save_results_completion_permutations_{len}'
    if mode == 'scrambled':
        save_name += '_scrambled'    

    with open(f'quality_metrics/gpt_judgements/{save_name}.json', 'w') as f:
        json.dump(data, f)


def test_permutations(N=100):
    client, cfg_generation = get_default_config()
    list_outputs = []

    for i in range(N):
        print(f'Step {i}')
        completion = get_completion(client, random_permutation_prompt, cfg_generation)
        m = re.match('.*(\[.*\])(.*)', completion)
        if m is None:
            print('No match for list')
            print(f'Completion is: {completion}')
            continue
        else:
            string_list = m[1]

        try:
            result = eval(string_list)
        except Exception as e:
            print("Evaluation failed with exception:")
            print(e)
            continue

        if not isinstance(result, list):
            print(f"The result doesn't have the required type (it's a {type(result)})")
            continue
        
        if len(result) != 5:
            print(f"The result doesn't have the required length ({len(result)})")
            continue
        
        list_outputs.append((i, result))
    
    with open('quality_metrics/gpt_judgements/chatgpt_rng.json', 'w') as f:
        json.dump(list_outputs, f)


# perform tests
if __name__ == "__main__":
    client, cfg_generation = get_default_config()
    # print(get_five_fold_results(client, cfg_generation))
    test_validity(1000, mode='scrambled', puzzle_set='preferences')
    # test_permutations(1000)