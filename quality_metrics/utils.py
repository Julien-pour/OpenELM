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
    body = np.random.choice(puzzle['sol_bodies'])
    return ''.join([header, body])


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
