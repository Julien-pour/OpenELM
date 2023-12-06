import gc
import os
import pathlib
import json
import torch

from torch.optim import SGD
from quality_metrics import utils

from peft import get_peft_model, LoraConfig, TaskType


def get_cross_entropy(model, input_ids, attention_mask):
    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # average non-masked tokens over seq dim
    loss = loss.view(batch_size, seq_len - 1)
    loss = (loss * attention_mask[..., :-1].contiguous()).sum(-1) / attention_mask.sum(-1)

    return loss


@torch.no_grad()
def get_puzzle_solution_likelihoods():
    return 0.


@torch.no_grad()
def get_solution_logprobs(tokenized_puzzle_archive, model, batch_size=2):
    # todo should cut batches to remove unnecessary tokens
    #   + check that logprob computation is ok / get perplexity computation somewhere
    all_losses = []
    for i in range(0, tokenized_puzzle_archive.input_ids.shape[0], batch_size):
        input_ids = tokenized_puzzle_archive.input_ids[i:i+batch_size].to(model.device)
        attention_mask = tokenized_puzzle_archive.attention_mask[i:i+batch_size].to(model.device)

        loss = get_cross_entropy(model, input_ids, attention_mask)

        # nll = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.clone()).loss

        # todo: check these calculations, maybe they're more indicative of compression progress?
        # model_logits = model(input_ids, attention_mask).logits  # check
        # ll = model_logits.view(model_logits.shape[0] * model_logits.shape[1], model_logits.shape[2])
        # norm = model_logits.logsumexp(-1)
        # ii = input_ids.view(input_ids.shape[0] * input_ids.shape[1])
        # logits = ll[torch.arange(ii.shape[0]).to(input_ids.device), ii].view(batch_size, -1)
        # logprobs = logits - norm
        # logprobs = (attention_mask * logprobs).sum(-1) / attention_mask.sum(-1)  # average over sequence dim

        pass
        all_losses.append(loss.cpu())
    return torch.cat(all_losses, dim=0)


# optimizer must be not have momentum
def get_compression_progress(tokenized_puzzle, tokenized_puzzle_archive, model, optimizer,
                             original_losses=None):
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


def eval_compression_progress(puzzles_to_test_path, puzzle_archive_path, model_id, prompt_path=None,
                              save_name='likelihood_save_file', save_dir='logs/compression_progress_test'):
    # load puzzles
    puzzles = json.load(open(puzzles_to_test_path, 'r'))
    puzzle_archive = json.load(open(puzzle_archive_path, 'r'))

    # create model and tokenizer
    model, tokenizer = utils.create_model_and_tokenizer(model_id, compile=False)

    # lora
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
    puzzle_strs = [p['sat'].replace('def sat(', 'def f(') for p in puzzles if p['sol_bodies']]
    sol_strs = [utils.make_solution(p) for p in puzzles if p['sol_bodies']]
    if prompt_text is not None:
        puzzle_sols = [prompt_text.format(puzzle=puz, solution=sol) for puz, sol in zip(puzzle_strs, sol_strs)]
    else:
        puzzle_sols = [f"Puzzle:\n\n{puz}\n\nSolution:\n\n{sol}" for puz, sol in zip(puzzle_strs, sol_strs)]
    tokenized_puzzles = tokenizer(puzzle_sols, return_tensors='pt', padding=True)

    archive_puzzle_strs = [p['sat'].replace('def sat(', 'def f(') for p in puzzle_archive if p['sol_bodies']]
    archive_sol_strs = [utils.make_solution(p) for p in puzzle_archive if p['sol_bodies']]
    if prompt_text is not None:
        archive_puzzle_sols = [prompt_text.format(puzzle=puz, solution=sol) for puz, sol in
                               zip(archive_puzzle_strs, archive_sol_strs)]
    else:
        archive_puzzle_sols = [f"Puzzle:\n\n{puz}\n\nSolution:\n\n{sol}" for puz, sol in
                               zip(archive_puzzle_strs, archive_sol_strs)]
    archive_tokenized_puzzles = tokenizer(archive_puzzle_sols, return_tensors='pt', padding=True)

    # get difference in logprobs for each puzzle
    likelihood_diff_matrix = torch.zeros(tokenized_puzzles.input_ids.shape[0],
                                         archive_tokenized_puzzles.input_ids.shape[0])
    original_losses = get_solution_logprobs(archive_tokenized_puzzles, model)

    for i in range(tokenized_puzzles.input_ids.shape[0]):
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

    # save result matrix
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(likelihood_diff_matrix, os.path.join(save_dir, save_name))


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

    # simple test
    eval_compression_progress(puzzles_to_test, puzzle_path_archive, model_id)

    # expe todo: measure compression progress on the dev dataset for 1 opt step on each of the dev puzzles
    #       report the matrix for a range of learning rates
    ...
