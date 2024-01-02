import gc
import os
import pathlib

from datetime import datetime

import json
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from torch.optim import SGD
from quality_metrics import utils

from peft import get_peft_model, LoraConfig, TaskType


loss_fct = torch.nn.CrossEntropyLoss(reduce=False)


def get_cross_entropy(model, input_ids, attention_mask, loss_attention_mask=None):
    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # average non-masked tokens over seq dim
    loss = loss.view(batch_size, seq_len - 1)
    if loss_attention_mask is None:
        loss = (loss * attention_mask[..., :-1].contiguous()).sum(-1) / attention_mask.sum(-1)
    else:
        loss = (loss * loss_attention_mask[..., :-1].contiguous()).sum(-1) / loss_attention_mask.sum(-1)

    return loss


@torch.no_grad()
def get_puzzle_solution_likelihoods():
    return 0.


@torch.no_grad()
def get_solution_logprobs(tokenized_puzzle_archive, model, batch_size=2):
    if tokenized_puzzle_archive:
        try:
            mask = tokenized_puzzle_archive.loss_attention_mask
            mask_puzzle = True
        except AttributeError:
            mask_puzzle = False
    else:
        mask_puzzle = False

    all_losses = []
    for i in tqdm(range(0, tokenized_puzzle_archive.input_ids.shape[0], batch_size)):
        input_ids = tokenized_puzzle_archive.input_ids[i:i+batch_size].to(model.device)
        attention_mask = tokenized_puzzle_archive.attention_mask[i:i+batch_size].to(model.device)

        if not mask_puzzle:
            # use the loss over both puzzle and solution
            loss = get_cross_entropy(model, input_ids, attention_mask)
        else:
            # use loss over solution only
            loss_attention_mask = tokenized_puzzle_archive.loss_attention_mask[i:i+batch_size].to(model.device)
            loss = get_cross_entropy(model, input_ids, attention_mask, loss_attention_mask)

        all_losses.append(loss.cpu())
    return torch.cat(all_losses, dim=0)


# optimizer must be not have momentum
def get_compression_progress(tokenized_puzzle, tokenized_puzzle_archive, model, optimizer,
                             original_losses=None, ):
    # compute likelihood of solutions before
    if original_losses is None:
        original_losses = get_solution_logprobs(tokenized_puzzle_archive, model)

    # step on the current puzzle
    # todo: the memory costs seem to keep increasing here, try to fix
    model.train()
    optimizer.zero_grad()
    tokenized_puzzle.labels = tokenized_puzzle.input_ids.clone()
    loss = model(**tokenized_puzzle).loss
    loss.backward()
    optimizer.step()
    model.eval()

    # clear memory
    del loss
    del tokenized_puzzle.labels
    gc.collect()
    torch.cuda.empty_cache()

    # compute likelihood of solutions after
    final_losses = get_solution_logprobs(tokenized_puzzle_archive, model)
    differences = final_losses - original_losses
    return differences


def compression_progress_wrapper(prompt_text, puzzles, puzzle_archive, tokenizer, model, optimizer,
                                 use_docstring=False, mask_puzzle=True):
    # tokenizes the puzzles, and computes the finetuning compression progress metric on the
    # puzzles x archive matrix

    # make example for finetuning the model
    puzzle_strs = [utils.make_puzzle(p, use_docstring) for p in puzzles if p['sol_bodies']]
    sol_strs = [utils.make_solution(p) for p in puzzles if p['sol_bodies']]
    if prompt_text is not None:
        puzzle_sols = [prompt_text.format(puzzle=puz, solution=sol) for puz, sol in zip(puzzle_strs, sol_strs)]
    else:
        puzzle_sols = [f"Puzzle:\n```python\n{puz}\n```\nSolution:\n```python\n{sol}\n```"
                       for puz, sol in zip(puzzle_strs, sol_strs)]
    tokenized_puzzles = tokenizer(puzzle_sols, return_tensors='pt', padding=True)

    # make puzzle-sol pairs from the archive to measure if a given puzzle helps
    archive_puzzle_strs = [utils.make_puzzle(p, use_docstring) for p in puzzle_archive if p['sol_bodies']]
    archive_sol_strs = [utils.make_solution(p) for p in puzzle_archive if p['sol_bodies']]
    if prompt_text is not None:
        archive_puzzle_sols = [prompt_text.format(puzzle=puz, solution=sol) for puz, sol in
                               zip(archive_puzzle_strs, archive_sol_strs)]
    else:
        archive_puzzle_sols = [f"Puzzle:\n```python\n{puz}\n```\nSolution:\n```python\n{sol}\n```"
                               for puz, sol in zip(archive_puzzle_strs, archive_sol_strs)]
    archive_tokenized_puzzles = tokenizer(archive_puzzle_sols, return_tensors='pt', padding=True)

    # if we only compute compression progress on the solution, get the mask
    if mask_puzzle:
        solutions_tokenized = tokenizer(archive_sol_strs)
        solution_attention_mask = torch.zeros_like(archive_tokenized_puzzles.attention_mask)
        # compute the solution attention mask
        for idx, (full_prompt, sol) in enumerate(zip(archive_tokenized_puzzles.input_ids,
                                                     solutions_tokenized.input_ids)):
            mask = utils.get_solution_mask(full_prompt, sol)
            solution_attention_mask[idx] = mask
        archive_tokenized_puzzles.loss_attention_mask = solution_attention_mask

    # get difference in logprobs for each puzzle:
    # first initialize
    likelihood_diff_matrix = torch.zeros(tokenized_puzzles.input_ids.shape[0],
                                         archive_tokenized_puzzles.input_ids.shape[0])
    # then get original losses
    original_losses = get_solution_logprobs(archive_tokenized_puzzles, model)

    for i in tqdm(range(tokenized_puzzles.input_ids.shape[0])):
        tokenized_puzzle = utils.AttrDict(
            input_ids=tokenized_puzzles.input_ids[i:i+1].to(model.device),
            attention_mask=tokenized_puzzles.attention_mask[i:i+1].to(model.device),
        )

        tokenized_puzzle = utils.remove_unnecessary_indices(tokenized_puzzle)

        likelihood_diff_matrix[i] = get_compression_progress(
            tokenized_puzzle,
            archive_tokenized_puzzles,
            model,
            optimizer,
            original_losses,
        ).cpu()

    return likelihood_diff_matrix


### in context compression progress


def get_in_context_compression_progress(archive_tokenized_puzzles, archive_tokenized_puzzles_with_example,
                                        model, original_losses):
    # the archive tokenized puzzles might not be necessary if we have the original loss

    # compute likelihood of solutions before
    if original_losses is None:
        original_losses = get_solution_logprobs(archive_tokenized_puzzles, model)

    final_losses = get_solution_logprobs(archive_tokenized_puzzles_with_example, model)
    differences = final_losses - original_losses
    return differences


def incontext_compression_progress_wrapper(prompt_text, puzzles, puzzle_archive, tokenizer, model,
                                           use_docstring=False, mask_puzzle=True):
    # tokenizes the puzzles, and computes the finetuning compression progress metric on the
    # puzzles x archive matrix

    # Note: must have a prompt that allows in-context examples
    assert prompt_text is not None

    # make the prompts to measure baseline likelihood of solution in the archive
    archive_puzzle_strs = [utils.make_puzzle(p, use_docstring) for p in puzzle_archive if p['sol_bodies']]
    archive_sol_strs = [utils.make_solution(p) for p in puzzle_archive if p['sol_bodies']]
    if use_docstring:
        ref_puzzle = utils.REF_PUZZLE
    else:
        ref_puzzle = utils.REF_PUZZLE_NODOC
    ref_solution = utils.REF_SOL
    archive_puzzle_sols = [prompt_text.format(puzzle=ref_puzzle, solution=ref_solution,
                                              archive_puzzle=apuz, archive_solution=asol) for apuz, asol in
                           zip(archive_puzzle_strs, archive_sol_strs)]
    archive_tokenized_puzzles = tokenizer(archive_puzzle_sols, return_tensors='pt', padding=True)
    # if we only get the loss on the solution, compute the solution masks
    if mask_puzzle:
        solutions_tokenized = tokenizer(archive_sol_strs)
        solution_attention_mask = torch.zeros_like(archive_tokenized_puzzles.attention_mask)
        # compute the solution attention mask
        for idx, (full_prompt, sol) in enumerate(zip(archive_tokenized_puzzles.input_ids,
                                                     solutions_tokenized.input_ids)):
            mask = utils.get_solution_mask(full_prompt, sol)
            solution_attention_mask[idx] = mask
        archive_tokenized_puzzles.loss_attention_mask = solution_attention_mask

    # make the prompts to measure how much a given puzzle helps on solving the archive puzzles
    puzzle_strs = [utils.make_puzzle(p, use_docstring) for p in puzzles if p['sol_bodies']]
    sol_strs = [utils.make_solution(p) for p in puzzles if p['sol_bodies']]
    archive_puzzle_sols_with_example = []
    for i, (puz, sol) in enumerate(zip(puzzle_strs, sol_strs)):
        archive_puzzle_sols_with_example.append([])
        for apuz, asol in zip(archive_puzzle_strs, archive_sol_strs):
            ins = {'puzzle': puz, 'solution': sol, 'archive_puzzle': apuz, 'archive_solution': asol}
            archive_puzzle_sols_with_example[i].append(prompt_text.format(**ins))

    # get difference in logprobs for each puzzle
    likelihood_diff_matrix = torch.zeros(len(puzzle_strs),
                                         len(archive_puzzle_strs))
    original_losses = get_solution_logprobs(archive_tokenized_puzzles, model)

    for i in tqdm(range(len(puzzle_strs))):
        # tokenized list of all archive puzzles prefixed by the example
        archive_tokenized_puzzles_with_example = tokenizer(archive_puzzle_sols_with_example[i], return_tensors='pt',
                                                           padding=True)

        # if we only get the loss on the solution, compute the archive solution masks
        if mask_puzzle:
            solutions_tokenized = tokenizer(archive_sol_strs)
            solution_attention_mask = torch.zeros_like(archive_tokenized_puzzles_with_example.attention_mask)
            # compute the solution attention mask
            for idx, (full_prompt, sol) in enumerate(zip(archive_tokenized_puzzles_with_example.input_ids,
                                                         solutions_tokenized.input_ids)):
                mask = utils.get_solution_mask(full_prompt, sol)
                solution_attention_mask[idx] = mask
            archive_tokenized_puzzles_with_example.loss_attention_mask = solution_attention_mask

        likelihood_diff_matrix[i] = get_in_context_compression_progress(
            archive_tokenized_puzzles,
            archive_tokenized_puzzles_with_example,
            model,
            original_losses,
        ).cpu()

    return likelihood_diff_matrix


def eval_compression_progress(puzzles_to_test_path, puzzle_archive_path, model_id, prompt_path=None,
                              save_name='progress_results', save_dir='logs/compression_progress_test',
                              analyze_compression_progress=True, use_lora=True, in_context=False,
                              file_prefix=None, use_docstring=False):

    if file_prefix is None:
        file_prefix = model_id.split('/')[-1] + '-' + str(datetime.now()).split()[0]

    # load puzzles
    puzzles = json.load(open(puzzles_to_test_path, 'r'))
    puzzle_archive = json.load(open(puzzle_archive_path, 'r'))

    # filter puzzles with no solution
    puzzles = [p for p in puzzles if p['sol_bodies']]
    puzzle_archive = [p for p in puzzle_archive if p['sol_bodies']]
    # todo remove when we have appropriate checks on puzzle len
    puzzles = [p for p in puzzles if len(p['sat']) + len(p['sol_bodies'][0]) < 1000]
    puzzle_archive = [p for p in puzzle_archive if len(p['sat']) + len(p['sol_bodies'][0]) < 1000]

    # create model and tokenizer
    model, tokenizer = utils.create_model_and_tokenizer(model_id, compile=False)

    # lora
    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    # create stateless optimizer
    optimizer = SGD(model.parameters(), lr=1e-04)  # todo change lr and stuff

    # tokenized prompt
    if prompt_path is not None:
        prompt_text = open(prompt_path, 'r').read()
    else:
        prompt_text = None

    # create tokenized archives and puzzle list (on cpu)
    # todo should filter puzzles with too many tokens
    if not in_context:
        likelihood_diff_matrix = compression_progress_wrapper(
            None,  # prompt text differs from the in context one, handle this somewhere
            puzzles,
            puzzle_archive,
            tokenizer,
            model,
            optimizer,
            use_docstring=use_docstring,
        )
    else:
        likelihood_diff_matrix = incontext_compression_progress_wrapper(
            prompt_text,
            puzzles,
            puzzle_archive,
            tokenizer,
            model,
            use_docstring=use_docstring,
        )

    # save result matrix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # torch.save(likelihood_diff_matrix, os.path.join(save_dir, save_name))

    # save json of results
    results = {}
    results['progress_type'] = 'finetuning' if not in_context else 'in_context'
    results['progress_metric'] = 'compression'  # todo add task lp when we have it
    results['tested_puzzle_names'] = [p['name'] for p in puzzles]
    results['tested_puzzles'] = [p['sat'] for p in puzzles]
    results['tested_sols'] = [utils.make_solution(p) for p in puzzle_archive]
    results['archive_puzzle_names'] = [p['name'] for p in puzzle_archive]
    results['archive_puzzles'] = [p['sat'] for p in puzzle_archive]
    results['archive_sols'] = [utils.make_solution(p) for p in puzzle_archive]
    results['compression_progress'] = likelihood_diff_matrix.tolist()

    json.dump(results, open(os.path.join(save_dir, file_prefix + '_' + save_name + '.json'), 'w'))

    if analyze_compression_progress:
        eval_dict = {}
        # get mean, std, histogram, print examples with their compression (maybe the extremes)
        compression_means = likelihood_diff_matrix.mean()
        compression_std = likelihood_diff_matrix.std()
        print(f"compression mean: {compression_means}")
        print(f"compression std: {compression_std}")

        puzzle_average_compression = likelihood_diff_matrix.mean(-1)

        # print lowest 3 puzzles and

        # histogram with all compressions
        plt.hist(puzzle_average_compression.tolist())
        plt.show()
        pass


def get_solution_completion_logprob(tokenized_puzzle_archive):
    pass


def get_task_learning_progress(tokenized_puzzle, tokenized_puzzle_archive, tokenizer, model, optimizer,
                               tokenized_prompt):
    pass


# test
if __name__ == "__main__":
    puzzle_path_archive = "puzzles_dev.json"
    puzzles_to_test = "puzzles_dev.json"
    model_id = "openlm-research/open_llama_3b_v2"

    prompt_path = "quality_metrics/dataset_progress/progress_base_example_prompt.md"

    # simple test, finuning based compression progress
    # eval_compression_progress(puzzles_to_test, puzzle_path_archive, model_id, prompt_path=prompt_path)

    # simple test, example-based compression progress
    eval_compression_progress(puzzles_to_test, puzzle_path_archive, model_id, prompt_path=prompt_path,
                              in_context=True, use_docstring=True)

    # expe todo: measure compression progress on the dev dataset for 1 opt step on each of the dev puzzles
    #       report the matrix for a range of learning rates
    ...
