from abc import ABC, abstractmethod
from typing import Tuple 
from pairrm import extract_single_rating_autoj, extract_pairwise_result_autoj, build_autoj_input
from tqdm import tqdm
import torch

import time

# Rank class
class Rank_puzzle(ABC):
    def __init__(self,puzzle_dict,mode_rank="pairwise",prompt_instruction=None, n_generation=4):
        """ 
        Args:
        - puzzle_dict: a dictionary of puzzles to rank
        - mode_rank: the mode to rank the puzzles, either "pairwise" or "absolute"
        - prompt_instruction: the prompt to use for the ranking
        - n_generation: the number of time to do pairwise ranking on a pair of puzzles or absolute ranking of a puzzle
        """
        
        self.prompt_instruction = prompt_instruction
        self.mode_rank = mode_rank
        self.puzzle_dict = puzzle_dict
        self.save_results = []
        self.save_results_inverse = []
        self.speed_inference = None
        self.list_speed_inference = []
        self.n_generation = n_generation
        self.save_all_results = []
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
        grades = {key: [] for key in self.puzzle_dict}
        with tqdm(total=int(len(keys)*self.n_generation)) as pbar:
            for i in range(len(keys)):
                for _ in range(self.n_generation):
                    key = keys[i]
                    puzzle = self.puzzle_dict[key]
                    grades[key].append(self.absolute_grade(puzzle))

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
        total_iter = int(len(keys)*(len(keys)-1)/2*self.n_generation)
        with tqdm(total=total_iter) as pbar:
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    puzzle1, puzzle2 = self.puzzle_dict[key1], self.puzzle_dict[key2]
                    
                    # Determine the winner using the pairwise ranking function
                    for _ in range(self.n_generation):
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
                        self.save_all_results.append((key1,key2,res_pairwise))
                        self.save_all_results.append((key2,key1,res_pairwise_inverse))
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
    
# HF model

class HF_Rank(Rank_puzzle):
    def __init__(self, puzzle_dict,prompt_instruction,mode_rank="pairwise",exllama2=True,model_id=None,revision="main",n_generation=4) -> None:
        """
        Args:
        - puzzle_dict: a dictionary of puzzles to rank
        - prompt_instruction: the prompt to use for the ranking
        - exllama2: whether to use exllama2
        - model_id: the model_id to use
        - revision: the revision to use
        - n_generation: the number of time to do pairwise ranking on a pair of puzzles or absolute ranking of a puzzle

        kwargs:
        - mode_rank: the mode to rank the puzzles, either "pairwise" or "absolute"

        """
        self.exllama2 = exllama2
        self.model_id = model_id
        self.revision = revision
        n_generation
        super().__init__(puzzle_dict=puzzle_dict,prompt_instruction=prompt_instruction,mode_rank=mode_rank,n_generation=n_generation)

    def init_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
        path_model=self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(path_model,revision = self.revision)
        if self.exllama2:
            gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
            self.model = AutoModelForCausalLM.from_pretrained(path_model,device_map="auto",quantization_config=gptq_config,revision = self.revision)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path_model,device_map="auto",revision = self.revision)
        self.model=torch.compile(self.model)

    def generate(self,text):
        with torch.no_grad():
            time_s = time.time()
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            out_tok = self.model.generate(**inputs, max_length=2048, do_sample=True, temperature = 1., top_p=0.9)
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
    
# evaluate with GAIR/autoj-13b-GPTQ-4bits 

class Auto_j_Rank(HF_Rank):
    def __init__(self, puzzle_dict,mode_rank="pairwise",prompt_instruction=None,exllama2=True,model_id="GAIR/autoj-13b-GPTQ-4bits") -> None:
        self.exllama2 = exllama2
        super().__init__(puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id)
        
        
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
    
# TheBloke/openchat-3.5-1210-GPTQ
prompt_openchat = """GPT4 Correct User: {instruct}<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:"""
instruction_openchat="""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback:
"""
to_rem="""
###Reference Answer (Score 5):
{orig_reference_answer}
"""
instruction_openchat_wo_reference = instruction_openchat.replace(to_rem,"")
# TODO: define description

class Open_chat(HF_Rank):
    def __init__(self, puzzle_dict,mode_rank="absolute",prompt_instruction=None,exllama2=True,model_id="TheBloke/openchat-3.5-1210-GPTQ",revision="gptq-4bit-32g-actorder_True", n_generation=4) -> None:
        self.exllama2 = exllama2
        super().__init__(model_id,puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,revision=revision, n_generation = n_generation)
    
    def absolute_grade(self,puzzle):
        """return the absolute_grade float between 0 and 5"""
        query = self.prompt_instruction
        resp1 = puzzle
        instruct = instruction_openchat.format(orig_instruction=query,orig_response=resp1,orig_reference_answer="...",orig_criteria="...",
                                    orig_score1_description="...",orig_score2_description="...",orig_score3_description="...",
                                    orig_score4_description="...",orig_score5_description="...")
        input_single = prompt_openchat.format(instruct=instruct)
        out = self.generate(input_single)
        raise NotImplementedError # need to extract the score
        return  