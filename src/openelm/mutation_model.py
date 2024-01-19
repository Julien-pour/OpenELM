import functools
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
#need to remove all langchain dependencies
# from langchain.chat_models import ChatOpenAI
# from langchain.llms.base import LLM
# from langchain.schema import Generation#, LLMResult
from concurrent.futures import ThreadPoolExecutor

class LLMResult:
    """Result of a language model generation. to replace langchain.schema.LLMResult"""
    def __init__(self, generations):
        self.generations = generations

# from langchain.schema.messages import HumanMessage

from pydantic import Extra, root_validator
from transformers import BatchEncoding

from openelm.codegen import model_setup, set_seed, truncate
from openelm.configs import ModelConfig
from openelm.utils.diff_eval import apply_diff, split_diff
from joblib import Parallel, delayed
from openai import OpenAI



def get_model(config: ModelConfig):
    if config.model_type == "hf":
        return HuggingFaceLLM(config=config)
    elif config.model_type == "openai":
        # Adapt config here
        # cfg: dict = {
        #     "temperature": config.temp,
        #     "top_p": config.top_p,
        #     # TODO: rename config option?
        #     "model_name": config.model_path,
        #     "timeout": config.request_timeout,
        # }

        # if config.gen_max_len!=-1:
        #     cfg["max_tokens"]=config.gen_max_len
        return OpenAI(max_retries=config.max_retries,timeout=config.request_timeout)#ChatOpenAI(**cfg)

    else:
        raise NotImplementedError

def get_completion(client, prompt : str, cfg_generation, tools=None,temperature=None)->str:
    """Get completion from OpenAI API"""
    kwargs={}
    kwargs.update(cfg_generation)
    if temperature is not None:
        kwargs["temperature"]= temperature
    flag_tool=tools is not None
    if flag_tool:
        kwargs.update({"tools": tools})
        tool_name=tools[0]["function"]["name"]
        kwargs.update({"tool_choice": {"type": "function", "function": {"name": tool_name}}})
    try :
        completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI programming assistant"},#You are a coding assistant, skilled in writting code with creative flair."},
            {"role": "user", "content": prompt}
        ],**kwargs
        )
    except Exception as e:
        print("completion problem: ",e)
        return None 
    # completion_token = completion.usage.completion_tokens
    # prompt_token = completion.usage.prompt_tokens
    
    if flag_tool:
        try:
            tool_out=out.choices[0].message.tool_calls[0].function.arguments
            return eval(tool_out)
        except Exception as e:  
            print("tool parsing problem: ",e)
            return None
        
    out = completion.choices[0].message.content
    return out

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def get_multiple_completions(client, batch_prompt: list[str], cfg_generation: dict, batch_tools: list[list[dict]]=None,max_workers=20,temperature=None)->list[str]:
    """Get multiple completions from OpenAI API
    batch_tools =[[tools]] tools is the function, toll_name is the name of the tool

                    /!\ need to integrate batch tools in the loop /!\
    """
    # check that batch_prompt is list[str]
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    completions = []
    if max_workers>1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch in chunks(batch_prompt, max_workers):
                for idx,message_list in enumerate(sub_batch):
                    # kwargs_modified = args.copy()
                    # kwargs_modified["messages"] = message_list
                    kwargs = {"client":client, "prompt":message_list}
                    kwargs["cfg_generation"]=cfg_generation
                    if temperature is not None:
                        kwargs["temperature"]= temperature
                    # if "kwargs" in kwargs_modified:
                    #     original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(
                        get_completion,**kwargs
                    )
                    completions.append(future)
        # Retrieve the results from the futures
        results = [future.result() for future in completions]
    else:
        for idx,message_list in enumerate(batch_prompt):
            # kwargs_modified = args.copy()
            # kwargs_modified["messages"] = message_list
            kwargs = {"client":client, "prompt":message_list}
            kwargs["cfg_generation"]=cfg_generation
            if temperature is not None:
                kwargs["temperature"]= temperature
            # if "kwargs" in kwargs_modified:
            #     original_kwargs = kwargs_modified.pop("kwargs")
            result = get_completion(**kwargs)
            completions.append(result)
            results = completions

    return results

class MutationModel(ABC):
    """Base model class for all mutation models."""

    def __init__(self) -> None:
        self.config: ModelConfig

    @abstractmethod
    def generate_programs(self, *args, **kwargs) -> list[str]:
        raise NotImplementedError


class PromptModel(MutationModel):
    """Mutation model that uses prompts to change a seed."""

    def __init__(self, config: ModelConfig) -> None:
        self.config: ModelConfig = config
        seed: int = set_seed(self.config.seed)
        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=seed)
        self.model = get_model(self.config)
        self.cfg_generation: dict = {
            "temperature": self.config.temp,
            "top_p": self.config.top_p,
            "model": self.config.model_path,
        }
        if self.config.gen_max_len != -1:
            self.cfg_generation["max_tokens"] = self.config.gen_max_len
    
    def generate_completion(self,list_prompt: list[str],batch_tools=None,temperature=None,activate_parrallel=True) -> list[str]:
        if "3.5" in self.config.model_path or "gpt-4" in self.config.model_path:
            if self.config.parrallel_call and activate_parrallel:
                # results = Parallel(n_jobs=self.config.processes)(delayed(self.model.generate)([[HumanMessage(content=prompt)]]) for prompt in prompts)
                results = get_multiple_completions(self.model, list_prompt, self.cfg_generation, batch_tools=batch_tools,max_workers=self.config.processes,temperature=temperature)
            else:
                results = get_multiple_completions(self.model, list_prompt, self.cfg_generation, batch_tools=batch_tools,max_workers=1,temperature=temperature)
        else: raise NotImplementedError
        return results
    
    def generate_programs(
        self,
        prompt_dicts: list[dict[str, str]],
        local_scope_truncate: bool,
        do_trunc=True,
        batch_tools=None,
        **kwargs
    ) -> list[str]:
        """
        Generate new programs from a batch of programs.

        Given a piece of code, do prompt mutation, execute the code,
        and return the result.

        Args:
            prompt_dicts (list[dict[str, str]): A list of dictionaries containing
            the prompt and template for each program.
            templates can be for example in P3: from typing import*\n\n
            local_scope_truncate (bool): Whether or not to truncate the code to
            the local scope.

        Returns:
            A list of code strings.
        """
        prompts = [prompt_dict["prompt"] for prompt_dict in prompt_dicts]
        templates = [prompt_dict["template"] for prompt_dict in prompt_dicts]
        if "3.5" in self.config.model_path or "gpt-4" in self.config.model_path:
            
            if self.config.parrallel_call:
                # results = Parallel(n_jobs=self.config.processes)(delayed(self.model.generate)([[HumanMessage(content=prompt)]]) for prompt in prompts)
                results = get_multiple_completions(self.model, prompts, self.cfg_generation, batch_tools=batch_tools,max_workers=self.config.processes)
            else:
                results = get_multiple_completions(self.model, prompts, self.cfg_generation, batch_tools=batch_tools,max_workers=1)
            completions = results
        else:
            results = self.model.generate(prompts=prompts)
            completions = [
                gen.text for sublist in results.generations for gen in sublist
            ]
        # Flatten nested list of generations

        if do_trunc:
            trunc = functools.partial(truncate, only_local_scope=local_scope_truncate)
            truncations: list[str] = [
                templates[i] + trunc(completions[i]) for i in range(len(completions))
            ]
        else:
            truncations: list[str] = [
                templates[i] + "\n    " + completions[i]
                for i in range(len(completions))
            ]

        return truncations


class DiffModel(PromptModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def generate_programs(
        self, prompt_dicts: list[dict[str, str]], local_scope_truncate: bool, **kwargs
    ) -> list[str]:
        # local_scope_truncate = False
        prompts = [prompt_dict["prompt"] for prompt_dict in prompt_dicts]
        templates = [prompt_dict["template"] for prompt_dict in prompt_dicts]
        results: LLMResult = self.model.generate(prompts=prompts)
        # Flatten nested list of generations
        completions: list[str] = [
            gen.text for sublist in results.generations for gen in sublist
        ]

        end_of_diff = re.compile("\n[^ +-@]+")
        trunc = functools.partial(truncate, only_local_scope=local_scope_truncate)
        truncations: list[str] = [
            templates[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        outputs: list[str] = []
        for i, code in enumerate(truncations):
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed: dict = split_diff(code)
            # truncate the diff hunk at the first line not starting with " ",
            # "+", "-", or "@".
            if parsed and all(
                (s in parsed for s in ["name", "file", "message", "diff"])
            ):
                diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
                nme_idx: int = diff_hunk.find("<NME>")
                if nme_idx != -1:
                    diff_hunk = diff_hunk[:nme_idx]
                outputs.append(apply_diff(prompts[i], diff_hunk))
        return outputs

class Generation:
    """replacement for langchain.schema.Generation"""
    def __init__(self, text, generation_info=None):
        self.text = text
        self.generation_info = generation_info
        self.type = "Generation"
        """Type is used exclusively for serialization purposes."""


class HuggingFaceLLM: #(LLM): <- removed langchain.llms.base.LLM inheritance need to check if it is still working
 
    config: ModelConfig
    model: Any = None
    tokenizer: Any = None
    device: Any = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    @root_validator
    def setup(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate the config."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if values["config"] is None:
            raise ValueError("Config must be provided.")
        if (
            values["model"] is None
            and values["tokenizer"] is None
            and values["device"] is None
        ):
            values["model"], values["tokenizer"], values["device"] = model_setup(
                values["config"]
            )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface"

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
        raise NotImplementedError

    def _generate(
        self, prompts: list[str], stop: Optional[list[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        batch_size = self.config.batch_size
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        encodings = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generations_dict: dict[str, list[Generation]] = defaultdict(list)

        for i in range(total_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(prompts))
            batched_prompts = BatchEncoding(
                {
                    "input_ids": encodings["input_ids"][start_index:end_index],
                    "attention_mask": encodings["attention_mask"][
                        start_index:end_index
                    ],
                }
            ).to(self.device)
            if self.config.logits_only:
                with torch.inference_mode():
                    outputs = self.model(**batched_prompts)
                    if i == 0:
                        logits = outputs.logits
                    else:
                        logits = torch.cat((logits, outputs.logits), dim=0)
                generations: list[Generation] = [
                    Generation(text="", generation_info={"logits": logits})
                    for logits in logits
                ]
            else:
                input_ids_len: int = batched_prompts["input_ids"].shape[1]
                with torch.inference_mode():
                    tokens = self.model.generate(
                        **batched_prompts,
                        do_sample=self.config.do_sample,
                        num_return_sequences=self.config.num_return_sequences,
                        temperature=self.config.temp,
                        max_new_tokens=self.config.gen_max_len,
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    texts: list[str] = self.tokenizer.batch_decode(
                        tokens[:, input_ids_len:, ...]
                    )
                generations = [Generation(text=text) for text in texts]

            for j, prompt in enumerate(prompts[start_index:end_index]):
                slice_start = j * self.config.num_return_sequences
                slice_end = slice_start + self.config.num_return_sequences
                generations_dict[prompt].extend(generations[slice_start:slice_end])

        return LLMResult(generations=list(generations_dict.values()))
