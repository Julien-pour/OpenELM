from abc import ABC, abstractmethod
from typing import Tuple 
from pairrm import extract_single_rating_autoj, extract_pairwise_result_autoj, build_autoj_input
from tqdm import tqdm
import torch

import time

# Rank class
class Rank_puzzle(ABC):
    def __init__(self,puzzle_dict,mode_rank="pairwise",prompt_instruction=None):
        self.prompt_instruction = prompt_instruction
        self.mode_rank = mode_rank
        self.puzzle_dict = puzzle_dict
        self.save_results = []
        self.save_results_inverse = []
        self.speed_inference = None
        
        self.list_speed_inference = []
        self.init_model()
        
    @abstractmethod
    def init_model(self):
        raise NotImplementedError
    
    def pairwise_ranking(self,puzzle1: str,puzzle2: str) -> int:
        """
        return the winner:
        - 0 if puzzle1 wins
        - 1 if puzzle2 wins
        - 2 if draw
        """
        raise NotImplementedError
    
    def absolute_grade(self,puzzle):
        """return the absolute_grade int or float"""
        raise NotImplementedError
    
    def absolute_ranking(self):
        """
        return the ranking of the puzzles
        """
        # Get a list of keys to iterate over
        keys = list(self.puzzle_dict.keys())
        grades = {key: 0 for key in self.puzzle_dict}
        with tqdm(total=len(keys)) as pbar:
            for i in range(len(keys)):

                key = keys[i]
                puzzle = self.puzzle_dict[key]
                grades[key] = self.absolute_grade(puzzle)

                pbar.update(1)
                if self.speed_inference!=None:
                    pbar.set_description(f"Speed Inference: {self.speed_inference} tok/s")
                    pbar.refresh()

        # Rank puzzles based on their win records, sorting by wins descending
        ranked_keys = sorted(keys, key=lambda x: grades[x], reverse=True)

        return ranked_keys, grades
    
    def round_robin_tournament(self,):
        # Initialize win records for each puzzle key
        win_record = {key: 0 for key in self.puzzle_dict}
        
        # Get a list of keys to iterate over
        keys = list(self.puzzle_dict.keys())
        
        # Iterate over each unique pair of puzzle keys
        total_iter = int(len(keys)*(len(keys)-1)/2)
        with tqdm(total=total_iter) as pbar:
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    puzzle1, puzzle2 = self.puzzle_dict[key1], self.puzzle_dict[key2]
                    
                    # Determine the winner using the pairwise ranking function
                    
                    res_pairwise = self.pairwise_ranking(puzzle1, puzzle2)
                    res_pairwise_inverse = self.pairwise_ranking(puzzle2, puzzle1)
                    

                    # Update the win record for the winner
                    if  res_pairwise == 0:
                        win_record[key1] += 1
                    elif res_pairwise == 1:
                        win_record[key2] += 1
                    elif res_pairwise == 2:
                        win_record[key1] += 0.5
                        win_record[key2] += 0.5
                    else: 
                        raise ValueError(f"Invalid result: {res_pairwise}")
                    
                    self.save_results.append((key1,key2,res_pairwise))
                    self.save_results_inverse.append((key2,key1,res_pairwise_inverse))
                    pbar.update(1)
                    if self.speed_inference!=None:
                        pbar.set_description(f"Speed Inference: {self.speed_inference} tok/s")
                        pbar.refresh()
                
        # Rank puzzles based on their win records, sorting by wins descending
        ranked_keys = sorted(keys, key=lambda x: win_record[x], reverse=True)
        
        # Convert ranked keys back to their corresponding puzzle names or descriptions
        ranked_puzzles = [(key, self.puzzle_dict[key]) for key in ranked_keys]
        return ranked_puzzles, win_record
    
    def computing_ranking(self) -> Tuple[list,dict]:
        if self.mode_rank == "pairwise":
            return self.round_robin_tournament()
        elif self.mode_rank == "absolute":
            return self.absolute_ranking()
        else:
            raise ValueError(f"Invalid ranking mode: {self.mode_rank}")
    

# evaluate with GAIR/autoj-13b-GPTQ-4bits 
        
class Auto_j_Rank(Rank_puzzle):
    def __init__(self, puzzle_dict,mode_rank="pairwise",prompt_instruction=None,exllama2=True) -> None:
        self.exllama2 = exllama2
        super().__init__(puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction)
        
    def init_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    

        path_model="GAIR/autoj-13b-GPTQ-4bits"
        self.tokenizer = AutoTokenizer.from_pretrained(path_model)
        if self.exllama2:
            gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
            self.model = AutoModelForCausalLM.from_pretrained(path_model,device_map="auto",quantization_config=gptq_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path_model,device_map="auto")
        self.model=torch.compile(self.model)
        
    def generate(self,text):
        with torch.no_grad():
            time_s = time.time()
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            out_tok = self.model.generate(**inputs, max_length=2048, temperature=0.0, do_sample=False, top_p=1.0)
            out = self.tokenizer.decode(out_tok[0], skip_special_tokens=True)
            time_e = time.time()
            speed = len(out_tok[0])/((time_e-time_s)) # result in tokens per second
            self.list_speed_inference.append(speed)
            if self.speed_inference==None:
                self.speed_inference =  speed
                # compute ema speed inference
            else:
                alpha = 0.2
                self.speed_inference = self.speed_inference
                self.speed_inference = (speed * alpha) + (self.speed_inference * (1-alpha))
        return out

    def pairwise_ranking(self,puzzle1: str,puzzle2: str) -> str:
        """return the winner (puzzle1 or puzzle2)"""
        query = self.prompt_instruction
        resp1 = puzzle1
        resp2 = puzzle2
        input_pairwise = build_autoj_input(prompt=query, 
                    resp1 = resp1,  resp2 = resp2, 
                    protocol = "pairwise_tie") # for pairwise response comparison 
        out = self.generate(input_pairwise)
        return extract_pairwise_result_autoj(out)
    
    def absolute_grade(self,puzzle):
        """return the absolute_grade float between 0 and 10"""
        query = self.prompt_instruction
        resp1 = puzzle
        input_single = build_autoj_input(prompt=query, 
                    resp1 = resp1, resp2=None, 
                    protocol = "single") # for single response evaluation 
        out = self.generate(input_single)
        return extract_single_rating_autoj(out)