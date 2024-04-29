"""
Experimental protocols to fix heritability.

Given a pool of individuals, a mutation operator and a set of metrics 
(eg embedding and fitness) the experiment reports correlation/similarity
between parents and offspring.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import textwrap
import hydra
from hydra import compose, initialize
from functools import partial
import json
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from typing import List
from typing import Optional

from pprint import pprint
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import torch

import os
os.environ['TRANSFORMERS_CACHE'] = "models"

from openelm import ELM
os.environ["TOKENIZERS_PARALLELISM"] = "True"
                        
os.environ["HYDRA_FULL_ERROR"] = "1"

from transformers import logging
logging.set_verbosity_error()  # avoid all FutureWarnings

from transformers import AutoTokenizer, AutoModel

from openelm.environments.p3.p3 import P3ProbSolResult
from openelm.configs import P3ProbSolChatEnvConfig
from openelm.algorithms.map_elites import Map


def load_genome(path: str):
    return json.load(open(path, 'r'))


def filter_genomes(genomes, model_id, max_seq_len=None):
    # remove very long genomes (we can't embed them)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if max_seq_len is None:
        max_seq_len = tokenizer.model_max_length

    tokenized_lengths = [len(tok) for tok in tokenizer(
        [gen['program_str'] for gen in genomes]).input_ids]
    return [gen for gen, tokl in zip(genomes, tokenized_lengths) if tokl < max_seq_len]


# def mutate_archive(old_genomes, elm):  # NOP
#     # new_genomes = []
#     m = elm.qd_algorithm.env.batch_size
#     new_genomes = []
#     config = P3ProbSolChatEnvConfig()  # check this
#     for i in range(len(old_genomes) // m):
#         batch = [(P3ProbSolResult(config=config, **gen), []) for gen in old_genomes[i*m:(i+1)*m]]
#         new_genomes.extend(elm.qd_algorithm.env.mutate(batch))
#     return new_genomes

def build_pb(pb):
    config = P3ProbSolChatEnvConfig()  # check this
    probsol = P3ProbSolResult(config=config, **pb)
    probsol.unique_id = pb['unique_id']
    probsol.idx_generation = pb['idx_generation']
    return probsol


def sample_examples_elm(elm, genomes):
    # sample fewshot examples (not including the example to mutate)
    num_fewshot = elm.config.qd.n_fewshot_examples
    examples = []

    for exid in range(num_fewshot):
        examples.append(build_pb(np.random.choice(genomes)))

    return examples


def mutate_archive(old_genomes, elm):
    batch_size = elm.qd_algorithm.env.batch_size
    all_new_genomes = []
    num_batches = (len(old_genomes) + batch_size - 1) // batch_size
    for batch_index in range(num_batches):
        print(f'Batch index {batch_index}')
        batch = []
        start_index = batch_index * batch_size
        batch = []
        end_index = min(start_index + batch_size, len(old_genomes))  # Ensure we do not go out of bounds

        for i in range(start_index, end_index):

            # select stuff
            few_shot = sample_examples_elm(elm, old_genomes)
            examples = few_shot + [build_pb(old_genomes[i])]
            batch.append((examples, []))
        all_new_genomes.extend(elm.qd_algorithm.env.mutate(batch))
        ...
    # mutate
    all_new_genomes = [gen.__to_dict__() for gen in all_new_genomes]
    mutated_ids = [gen['puzzles_id_fewshot'][-1] for gen in all_new_genomes]
    list_ids = [gen['unique_id'] for gen in old_genomes]
    new_genomes = []
    new_old_genomes = [] # create duplicate to match n examples created
    for gen in all_new_genomes:
        idx_old = gen['puzzles_id_fewshot'][-1]
        if idx_old in list_ids:
            new_old_genomes.append(old_genomes[list_ids.index(idx_old)])

        else:
            raise
            new_old_genomes.append(None)
    # for gen in old_genomes: #TODO changet that old_genomes -> all_new_genomes
    #     if gen['unique_id'] in mutated_ids:
    #         new_genomes.append(all_new_genomes[mutated_ids.index(gen['unique_id'])])
    #     else:
    #         new_genomes.append(None)  # mutation didn't work for this guy
    return new_old_genomes,all_new_genomes


def sequence_average(t, a):
    t = t * a.unsqueeze(-1)  # remove masked tokens
    sa = a.sum(1)  # get sequence-wise sum
    return (t.sum(1) / sa.unsqueeze(-1))  # average


def embed(texts, tokenizer, model, batch_size, device):
    with torch.inference_mode():
        embs = []
        print(f'Embedding {len(texts)} texts')
        for i in tqdm(range(0,len(texts),batch_size)):
            toks = tokenizer(
                texts[i:i+batch_size],
                return_tensors='pt',
                padding=True
            )
            outs = model(
                toks.input_ids.to(device),
                toks.attention_mask.to(device),
                output_hidden_states=True
            )
            embs.extend(sequence_average(outs.hidden_states[-1], toks.attention_mask.to(device)))
        return embs

def embed2(texts, model):
    with torch.inference_mode():
        embeddings=torch.tensor(model.encode(texts))
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings_norm


def pointwise_sim(vec1, vec2):
    cos=torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    res=cos(vec1, vec2)
    return res


@torch.no_grad()
def similarities(old_genomes, new_genomes, model_id, batch_size):
    # tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-code')
    if "jina-embeddings-v2-base-code" in model_id: 
        model = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v2-base-code', 
            trust_remote_code=True,
            load_in_8bit=True,
            device_map='auto'
        )

        old_embeddings = embed2(
            [gen['program_str'] for gen in old_genomes],
            model,
        )
        # old_embeddings_2 = embed2(
        #     [gen['program_str'] for gen in old_genomes],
        #     model,
        # )
        new_embeddings = embed2(
            [gen['program_str'] for gen in new_genomes],
            model, 
        )
    
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device)
        model.eval()

        old_embeddings = embed(
            [gen['program_str'] for gen in old_genomes],
            tokenizer,
            model,
            batch_size,
            device,
        )
        # old_embeddings_2 = embed2(
        #     [gen['program_str'] for gen in old_genomes],
        #     model,
        # )
        new_embeddings = embed(
            [gen['program_str'] for gen in new_genomes],
            tokenizer,
            model, 
            batch_size,
            device,
        )
    
        old_embeddings = torch.stack(old_embeddings, dim=0)
        old_embeddings /= torch.linalg.vector_norm(old_embeddings, dim=-1).unsqueeze(-1)  # normalize
        new_embeddings = torch.stack(new_embeddings, dim=0)
        new_embeddings /= torch.linalg.vector_norm(new_embeddings, dim=-1).unsqueeze(-1)  # normalize

    sim = pointwise_sim(old_embeddings, new_embeddings)
    return sim.cpu()


def quality_correlation(quality_metric, old_genomes, new_genomes):
    if quality_metric == None:
        old_fitness = [gen['fitness'] for gen in old_genomes]
        new_fitness = [gen['fitness'] for gen in new_genomes]
    
    return {"original_fitness":old_fitness,"new_fitness":new_fitness}


def get_metrics(quality_metric, old_genomes, new_genomes, model_id, batch_size):
    path_train="/home/flowers/work/OpenELM/src/openelm/utils/preprocess_p3_emb_dedup_puzzles.json"
    with open(path_train, 'r') as f:
        train_genomes = json.load(f)

    metric_dict = {}
    metric_dict['quality_correlation'] = quality_correlation(
        quality_metric,
        old_genomes,
        new_genomes
    )
    metric_dict['embedding_similarity'] = similarities(
        old_genomes,
        new_genomes,
        model_id,
        batch_size,
    ).tolist()
    metric_dict['embedding_similarity_scrambled'] = similarities(
        old_genomes,
        train_genomes[:len(old_genomes)],# np.random.permutation(new_genomes).tolist(),
        model_id,
        batch_size,
    ).tolist()

    print(f'Embedding similarity mean {np.mean(metric_dict["embedding_similarity"])}')
    print(f'Embedding similarity std {np.std(metric_dict["embedding_similarity"])}')
    print(f'Embedding similarity scrambled mean '
          f'{np.mean(metric_dict["embedding_similarity_scrambled"])}')
    print(f'Embedding similarity scrambled std '
          f' {np.std(metric_dict["embedding_similarity_scrambled"])}')
    # TODO add 
    print(metric_dict)
    return metric_dict


def base_elm_prompt_fn(
    list_few_shot_example : List[P3ProbSolResult],
    code_batch: Optional[List[P3ProbSolResult]] = None,
    skill_targeted: Optional[List[int]]=None,
    n_fewshot_ex=2,
    prompt: Optional[str] = None,
):
    puzzles = [puzz for puzz in list_few_shot_example[:n_fewshot_ex]]
    
    examples = ""
    for i, puzzle in enumerate(puzzles):   
        puzzle_description = puzzle.description # /!\ need to implement that puzzle.description /!\
        examples += f"\nPuzzle {i}:\nPuzzle description: {puzzle_description}\n```python\n{puzzle.program_str}\n```\n"
    puzzle_description = code_batch[0].description
    p_str = code_batch[0].program_str
    examples += f"\nPuzzle {i+1} (to mutate):\nPuzzle description: {puzzle_description}\n```python\n{p_str}\n```\n"

    default_prompt = """    You are a helpful assistant to a Professor teaching a programming course in Python.
    The Professor wants to give some puzzles to his master's students to teach them Python.

    I already have a series of Python Programming Puzzles (P3). Each puzzle consists of two functions: a problem function `f` and its corresponding solution `g`. The challenge lies in constructing a SAT problem `f` and a function `g` such that `f(g())` evaluates to `True`.
    I will provide two existing puzzles for reference, and I need you to create five new distinct puzzles (Puzzle 2 to Puzzle 6), each being a **mutation** derived from Puzzle {N}.
    
    Rules:
    - Each puzzle includes two functions: `def f(...)` and `def g(...)`.
    - The first argument of `f` is always the output from `g()`.
    - Ensure `f` and `g` have matching argument signatures (e.g., `def f(arg0, arg1=value1, arg2=value2, ...)` and `def g(arg1=value1, arg2=value2, ...)`).
    - Avoid using `f` inside `g`, and `g` inside `f`.
    - Include any necessary imports so your code runs smoothly.
    - Give a clear Puzzle description that must be brief and diverse compared to the other puzzles.
    - Make sure the puzzle is self-contained within these two functions.

    Puzzle Format:
    Puzzle description: A brief, one to two sentence summary of the puzzle's content.
    ```python
    def f(solution, args=...) -> bool:
        # Python code to test the solution returned by g.
        # This function is a test unit and must return True if the solution is correct, and False otherwise.

    def g(args=...) -> solution:
        # Python code to generate a solution for the problem.
        # The solution should generalize to all possible args.
        return solution

    assert f(g()) == True
    ```
    {examples}
    Your Task:
    Create five new Python Programming Puzzles (Puzzle 2 to Puzzle 6).
    Ensure that each new puzzle is created by making mutations to Puzzle {N}."""
    default_prompt=textwrap.dedent(default_prompt)
    if prompt is None:
        prompt = default_prompt

    prompt = prompt.replace('\n    ', '\n')
    prompt = prompt.format(examples=examples, N=i+1)
    return prompt


# @hydra.main(
#     config_name="elm_nlp",
#     version_base="1.2",
# )
def main(
    config_name: Optional[str] = "elm_nlp",
    prompt_to_test: Optional[str] = None,
    num_puz: Optional[int] = 100,
):
    
    # if config is None:  # hydra not initialized
    #     # fetch default config for elm
    with initialize(version_base="1.2"):
        config = compose(config_name=config_name)

    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    elm = ELM(config)
    elm.qd_algorithm.env.config.activate_filtering_description = False
    if prompt_to_test is not None:
        elm.qd_algorithm.env.prompt_seed_function = partial(base_elm_prompt_fn, prompt=prompt_to_test)
    if "elm" in config_name:
        elm.config.qd.n_fewshot_examples = elm.config.qd.n_fewshot_examples -1
    quality_metric = None
    # model_id = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    model_id = "/home/flowers/work/hf/deepseek-coder-1.3b-base"#'/home/flowers/work/hf/jina-embeddings-v2-base-code'

    old_genomes = load_genome('/home/flowers/work/OpenELM/analysis_P3/heritability/selected_genomes_umap.json')#"/home/flowers/work/OpenELM/src/openelm/utils/preprocess_p3_emb_dedup_puzzles.json")
    old_genomes = filter_genomes(old_genomes, model_id)
    old_genomes = old_genomes[:num_puz]

    # make number of genomes divisible by mutation batch size
    mutation_batch_size = elm.qd_algorithm.env.batch_size
    # last_index = len(old_genomes) - (len(old_genomes) % mutation_batch_size)
    # old_genomes = old_genomes[:last_index]
    
    old_genomes,new_genomes = mutate_archive(old_genomes, elm)
    old_genomes, new_genomes = zip(*[(old_gen, new_gen) 
        for old_gen, new_gen in zip(old_genomes, new_genomes) if new_gen is not None])
    metric_dict = get_metrics(
        quality_metric,
        old_genomes,
        new_genomes,
        model_id=model_id,
        batch_size=4,
    )
    print('done')

    with open('heritability_metrics.json', 'w') as f:
        json.dump(metric_dict, f)
    with open("puzzle.json", "w") as f:
        json.dump(new_genomes, f)


    return metric_dict
if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main()
    