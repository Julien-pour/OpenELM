"""
Experimental protocols to fix heritability.

Given a pool of individuals, a mutation operator and a set of metrics 
(eg embedding and fitness) the experiment reports correlation/similarity
between parents and offspring.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import hydra
import json
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from typing import List

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
    num_batches = len(old_genomes) // batch_size
    for batch_index in range(num_batches):
        print(f'Batch index {batch_index}')
        batch = []
        for i in range(batch_size):
            # select stuff
            few_shot = sample_examples_elm(elm, old_genomes)
            examples = few_shot + [build_pb(old_genomes[batch_index*batch_size+i])]
            batch.append((examples, []))
        all_new_genomes.extend(elm.qd_algorithm.env.mutate(batch))
        ...
    # mutate
    all_new_genomes = [gen.__to_dict__() for gen in all_new_genomes]
    mutated_ids = [gen['puzzles_id_fewshot'][-1] for gen in all_new_genomes]
    new_genomes = []
    for gen in old_genomes:
        if gen['unique_id'] in mutated_ids:
            new_genomes.append(all_new_genomes[mutated_ids.index(gen['unique_id'])])
        else:
            new_genomes.append(None)  # mutation didn't work for this guy
    return new_genomes


def sequence_average(t, a):
    t = t * a.unsqueeze(-1)  # remove masked tokens
    sa = a.sum(1)  # get sequence-wise sum
    return (t.sum(1) / sa.unsqueeze(-1))  # average


def embed(texts, tokenizer, model, batch_size, device):
    embs = []
    print(f'Embedding {len(texts)} texts')
    for i in tqdm(range(len(texts) // batch_size + 1)):
        toks = tokenizer(
            texts[i*batch_size:(i+1)*batch_size],
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
    return (vec1 * vec2).sum(-1)


@torch.no_grad()
def similarities(old_genomes, new_genomes, model_id, batch_size):
    # tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-code')
    # model = AutoModel.from_pretrained(
    #     'jinaai/jina-embeddings-v2-base-code', 
    #     trust_remote_code=True,
    #     load_in_8bit=True,
    #     device_map='auto'
    # )
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
    return 0.


def get_metrics(quality_metric, old_genomes, new_genomes, model_id, batch_size):
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
        np.random.permutation(new_genomes).tolist(),
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
    pprint(metric_dict)
    return metric_dict


@hydra.main(
    config_name="elm_nlp",
    version_base="1.2",
)
def main(config):
    path_out = HydraConfig.get().runtime.output_dir
    config.output_dir = path_out
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)
    config.qd.unique_id = config.unique_id+"_s"+str(config.qd.seed)+"_p"

    elm = ELM(config)
    # print(
    #     "Best Individual: ",
    #     elm.run(init_steps=config.qd.init_steps, total_steps=config.qd.total_steps),
    # )

    quality_metric = None
    model_id = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    # model_id = 'jinaai/jina-embeddings-v2-base-code'

    old_genomes = load_genome('selected_genomes_umap.json')
    old_genomes = filter_genomes(old_genomes, model_id)
    old_genomes = old_genomes[:10]  # dev

    # make number of genomes divisible by mutation batch size
    mutation_batch_size = elm.qd_algorithm.env.batch_size
    last_index = len(old_genomes) - (len(old_genomes) % mutation_batch_size)
    old_genomes = old_genomes[:last_index]
    
    new_genomes = mutate_archive(old_genomes, elm)
    old_genomes, new_genomes = zip(*[(old_gen, new_gen) 
        for old_gen, new_gen in zip(old_genomes, new_genomes) if new_gen is not None])
    metric_dict = get_metrics(
        quality_metric,
        old_genomes,
        new_genomes,
        model_id=model_id,
        batch_size=8,
    )
    print('done')

    with open('heritability_metrics.json', 'w') as f:
        json.dump(metric_dict, f)
    return metric_dict


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
    