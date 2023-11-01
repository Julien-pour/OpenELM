# add own difficulty computing script, todo integrate it with the rest
import os
import sys
import wandb
import torch
import copy
import ast
import numpy as np
import json
from difficulty.judge import test_puzzle, judge_parallel
from utils import pass_at_k, preprocessing_p3

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import List, Dict, Any, Union


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument('-m', '--model', default='openllama', choices=['openllama', 'codellama', 'opt', 'mistral7b'])
parser.add_argument('-d', '--dataset', default='dev', choices=['train', 'test', 'dev'])  # add generated datasets
parser.add_argument('-b', '--batch-size', default=2, type=int)  # add generated datasets
parser.add_argument('-p', '--process-number', default=-1, type=int)
parser.add_argument('-w', '--world-size', default=1, type=int)
parser.add_argument('--prompt-config', default=0, type=int)


MODEL_IDS = {
    "openllama": "openlm-research/open_llama_3b_v2",
    "codellama": "codellama/CodeLlama-7b-Python-hf",
    "mistral7b": "mistralai/Mistral-7B-v0.1",
    "opt": "facebook/opt-1.3b",
}


def split_samples(samples, world_size, process_number):
    num_samples = len(samples)
    divisor = num_samples // world_size
    remainder = num_samples % world_size
    if process_number < remainder:
        start_idx = process_number * (divisor + 1)
        end_idx = start_idx + divisor + 1
    else:
        over_remainder = process_number - remainder
        start_idx = remainder * (divisor + 1) + over_remainder * divisor
        end_idx = start_idx + divisor
    return samples[start_idx:end_idx]


def extract_sol_0(gen_text):
    # return text of the solution and text of the body
    if not gen_text.count('```') == 2:
        sol_text = gen_text.split('```')
        if not len(sol_text) >= 2:
            return ''
        try:
            to_unparse = [el for el in ast.parse(sol_text[1]).body if isinstance(el, ast.FunctionDef)]
            if not to_unparse:
                return ''
            sol_text = ast.unparse(to_unparse[0])
            return sol_text
        except SyntaxError:
            return ''
    else:
        sol_text = gen_text.split('```')[1]
        # parse the function
        try:
            to_unparse = [el for el in ast.parse(sol_text[1]).body if isinstance(el, ast.FunctionDef)]
            if not to_unparse:
                return ''
            sol_text = ast.unparse(to_unparse[0])
            return sol_text
        except SyntaxError:
            return ''


def extract_sol_1(gen_text):
    if not gen_text.count('assert f(g())'):
        return ''
    gen_text = gen_text.split('assert f(g())')[0].strip()
    try:
        parsed_fns = [p for p in ast.parse(gen_text).body if isinstance(p, ast.FunctionDef)]
    except SyntaxError:
        return ''
    if not parsed_fns:
        return ''
    else:
        return ast.unparse(parsed_fns[0])


PROMPT_CONFIGS = {
    0: {'prompt': 'solver_prompt_0.md', 'extract_sol': extract_sol_0},  # instruct-like prompt
    1: {'prompt': 'solver_prompt_1.md', 'extract_sol': extract_sol_1},
    2: {'prompt': 'solver_prompt_2.md', 'extract_sol': extract_sol_1},
    3: {'prompt': 'solver_prompt_3.md', 'extract_sol': extract_sol_1},
}


def eval_puzzle_loop(
        puzzle_path,
        model_id="facebook/opt-1.3b",  # "codellama/CodeLlama-7b-Python-hf"
        batch_size=2,
        num_return_sequences=10,
        out_file_name='eval_model',
        cur_idx=0,  # to resume
        stop_on_first_success=False,
        process_number=-1,
        world_size=1,
        prompt_config=0,
):
    # change this?
    # os.environ['HF_DATASETS_CACHE'] = "/projets/flowers/julien/hf/datasets"
    # os.environ['TRANSFORMERS_CACHE'] = "/projets/flowers/julien/models/"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )
    model.cuda()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    model = torch.compile(model)
    # print(model.hf_device_map)

    sys.set_int_max_str_digits(10_000)
    print('loading puzzles')
    with open(puzzle_path, mode='r') as f:
        puzzles = json.load(f)
    print('done')

    # files and paths, load puzzles
    out_file_name = '_'.join([out_file_name, model_id.split('/')[-1]])
    if process_number != -1:
        out_file_name += f'_p{process_number}'
        # only load part of the puzzles
        puzzles = split_samples(puzzles, world_size, process_number)

    print('preprocessing puzzles')
    all_puzzles = preprocessing_p3(puzzles, tokenizer=tokenizer)
    print('done')

    torch._dynamo.config.suppress_errors = True

    list_trainset = [[x["program_str"], x["g_firstline"]] for x in all_puzzles]
    curr_idx = 0
    list_passk = []
    list_passk_1 = []
    list_passk_2 = []
    list_passk_3 = []
    list_passk_4 = []
    list_passk_5 = []
    list_passk_6 = []
    list_passk_7 = []
    list_passk_8 = []
    list_passk_9 = []
    list_passk_10 = []

    list_puzzle = []
    probas_solved = []

    extract_sol = PROMPT_CONFIGS[prompt_config]['extract_sol']
    solver_prompt_path = os.path.join("difficulty", PROMPT_CONFIGS[prompt_config]['prompt'])
    solver_prompt = open(solver_prompt_path, 'r').read()

    print(f'Evaluating {len(list_trainset)} puzzles.')

    with torch.no_grad():
        for idx in tqdm(range(curr_idx, len(list_trainset), batch_size)):  # len(dataset["test"])

            print(f"\n\n============ idx {idx} ==================\n")
            list_prompt = []
            list_prompt_f = []
            list_prompt_g_firstline = []
            subset_train = list_trainset[idx:idx + batch_size]

            for (puzzle, g_firstline) in subset_train:
                prompt_f = puzzle.split("def g(")[0]
                list_prompt_g_firstline.append(g_firstline)
                list_prompt_f.append(prompt_f)
                prompt = solver_prompt.format(pb=prompt_f, g_firstline=g_firstline)
                list_prompt.append(prompt)
            inputs = tokenizer(list_prompt, return_tensors="pt", padding=True).to("cuda")
            # for idx_inp in range(len(inputs)):
            len_prompt = inputs["input_ids"].shape[1]
            list_puzzle_gen = [[] for _ in range(len(list_prompt))]

            for idx_gen in range(num_return_sequences):
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.8)
                generated_texts = tokenizer.batch_decode(outputs[:, len_prompt:], skip_special_tokens=True)
                for idx_out_gen in range(len(outputs)):
                    list_puzzle_gen[idx_out_gen].append(generated_texts[idx_out_gen])

            list_generated_text = copy.deepcopy(list_puzzle_gen)
            for i in range(len(list_puzzle_gen)):  # along the bs
                for j in range(len(list_puzzle_gen[i])):
                    prompt_f = list_prompt_f[i]
                    g_firstline = list_prompt_g_firstline[i]

                    print(list_puzzle_gen[i][j])
                    extract_g = extract_sol(list_puzzle_gen[i][j])
                    if extract_g:
                        extract_g = extract_g.splitlines()
                        extract_g[0] = g_firstline
                        extract_g = '\n'.join(extract_g)
                    extract_g = extract_g + "\n\nassert f(g()) == True"
                    test_fg = prompt_f + extract_g
                    list_puzzle_gen[i][j] = test_fg
                    list_puzzle.append(test_fg)
                    if j < 1:
                        print("\n-------------------\n")
                        print(test_fg)

                list_valid_puzzles = judge_parallel(list_puzzle_gen[i])

                cor_puz = np.sum(list_valid_puzzles)

                n_sample, n_correct = num_return_sequences, int(cor_puz)
                pass_k = pass_at_k(n_sample, n_correct, k=num_return_sequences)
                list_passk.append(pass_k)

                # todo make this depend on number of generated sequences
                list_passk_1.append(pass_at_k(n_sample, n_correct, k=1))
                list_passk_2.append(pass_at_k(n_sample, n_correct, k=2))
                list_passk_3.append(pass_at_k(n_sample, n_correct, k=3))
                list_passk_4.append(pass_at_k(n_sample, n_correct, k=4))
                list_passk_5.append(pass_at_k(n_sample, n_correct, k=5))
                list_passk_6.append(pass_at_k(n_sample, n_correct, k=6))
                list_passk_7.append(pass_at_k(n_sample, n_correct, k=7))
                list_passk_8.append(pass_at_k(n_sample, n_correct, k=8))
                list_passk_9.append(pass_at_k(n_sample, n_correct, k=9))
                list_passk_10.append(pass_at_k(n_sample, n_correct, k=10))

                proba_solved = n_correct / n_sample
                probas_solved.append(proba_solved)
                all_puzzles[idx + i]['proba_solved'] = proba_solved
                all_puzzles[idx + i]['n_sample'] = n_sample
                all_puzzles[idx + i]['n_correct'] = n_correct
                all_puzzles[idx + i]['generated_text'] = list_generated_text[i]
                all_puzzles[idx + i]['parsed_puzzles'] = list_puzzle_gen[i]

            print(f"correct puzzles: {int(np.sum(list_passk))}/{len(list_passk)}")
            with open(out_file_name + ".json", "w") as outfile:
                json.dump(list_passk, outfile)

        print(f"pass 1: {np.sum(list_passk_1)}/{len(list_passk)}")
        print(f"pass 3: {np.sum(list_passk_3)}/{len(list_passk)}")
        print(f"pass 5: {np.sum(list_passk_5)}/{len(list_passk)}")
        print(f"pass 7: {np.sum(list_passk_7)}/{len(list_passk)}")
        print(f"pass 10: {np.sum(list_passk_10)}/{len(list_passk)}")

        dic_passk = {"pass_1": float(np.sum(list_passk_1))}
        dic_passk["pass_2"] = float(np.sum(list_passk_2))
        dic_passk["pass_3"] = float(np.sum(list_passk_3))
        dic_passk["pass_4"] = float(np.sum(list_passk_4))
        dic_passk["pass_5"] = float(np.sum(list_passk_5))
        dic_passk["pass_6"] = float(np.sum(list_passk_6))
        dic_passk["pass_7"] = float(np.sum(list_passk_7))
        dic_passk["pass_8"] = float(np.sum(list_passk_8))
        dic_passk["pass_9"] = float(np.sum(list_passk_9))
        dic_passk["pass_10"] = float(np.sum(list_passk_10))

        # json_content = [dic_passk]
        # with open(out_file_name + "_e" + str(num_train_epochs) + ".json", "w") as outfile:
        #     json.dump(json_content, outfile, indent=4)
        json.dump(all_puzzles, open(out_file_name + '_puzzles_solved.json', 'w'))
        # wandb.log(dic_passk)


def eval_puzzles(puzzle_path: str, model_id: str):
    # given the path to a json, eval all puzzles
    # maybe modify to do parallel eval/do multiprocessing (afterwards)
    with open(puzzle_path, 'r') as f:
        all_puzzles = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,

        # quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model = torch.compile(model)



if __name__ == "__main__":
    args = parser.parse_args()
    model_id = MODEL_IDS[args.model]
    dataset_path = f"puzzles_{args.dataset}.json"

    eval_puzzle_loop(
        dataset_path,
        model_id=model_id,
        batch_size=args.batch_size,
        world_size=args.world_size,
        process_number=args.process_number,
        prompt_config=args.prompt_config
    )

    # wandb.finish()
