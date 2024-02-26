# class model to compute the ranking of a list of puzzles
import sys 
from openai import OpenAI
from utils_test_puzzle import get_completion

from abc import ABC, abstractmethod
from typing import Tuple 
from pairrm import extract_single_rating_autoj, extract_pairwise_result_autoj, build_autoj_input,build_Yes_input
from tqdm import tqdm
import torch

import time

# Rank class
class Rank_puzzle(ABC):
    def __init__(self,puzzle_dict,mode_rank="pairwise",prompt_instruction=None, n_generation=4):
        """ 
        Args:
        - puzzle_dict: a dictionary of puzzles to rank {puzzl_id: puzzle_text, ...}
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
    

class OpenAI_Rank(Rank_puzzle):
    def __init__(self, openai_key,puzzle_dict,prompt_instruction,mode_rank="absolute",model_id="gpt-3.5-turbo-0125",n_generation=1,temperature=0) -> None:
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
        self.temperature = temperature
        self.model_id = model_id
        self.openai_key = openai_key
        super().__init__(puzzle_dict=puzzle_dict,prompt_instruction=prompt_instruction,mode_rank=mode_rank,n_generation=n_generation)

    def init_model(self):
        self.cfg: dict = {
        "temperature": self.temperature,
        # "top_p": 1.,
        # TODO: rename config option?
        "model": self.model_id,
        "logprobs": False,
        # "top_logprobs": 5,
        "max_tokens": 200,
        }
        max_retries=10
        timeout=10
        self.client = OpenAI(api_key=self.openai_key,max_retries=max_retries, timeout=timeout)

    def generate(self,text):
        out = get_completion(self.client, text, self.cfg)
        return out
    

class OpenCodeInterpreter_1(OpenAI_Rank):
    def __init__(self, openai_key,puzzle_dict,mode_rank="absolute",prompt_instruction=None,model_id="gpt-3.5-turbo-0125",n_generation=1,temperature=0) -> None:
        self.prompt_1="""Rate the following code queries on a scale of 1 to 5 based on their complexity, where 1 is the easiest and 5 is the most
difficult. Consider the complexity of the query
Query: [{query}]
You are obliged to choose only from the following list.
Scoring Criteria:
1 Point - Very Basic: The query involves simple operations or common issues
2 Points - Basic: The query involves fundamental programming concepts or commonly used functions
3 Points - Intermediate: The query requires some programming experience, possibly involving multiple steps
4 Points - Difficult: The query involves advanced programming skills, including complex logic, algorithms, or data
structures
5 Points - Very Difficult: The query requires extensive expertise, potentially involving innovative problem-solving
approaches or unique algorithm design
Please give the score first with the format: "Score: [SCORE]" (write the score between bracket) then explain why"""
        super().__init__(openai_key=openai_key,puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,n_generation=n_generation,temperature=temperature)
        
    def absolute_grade(self,puzzle):
        """return the absolute_grade float between 0 and 10"""

        input_single = self.prompt_1.format(query=puzzle)
        n_try=3
        grade=-1
        while n_try>0:
            try:
                out = self.generate(input_single)
                grade=eval(out.split("[")[1].split("]")[0])
                assert grade in [1,2,3,4,5]
                return grade
            except:
                try:
                    grade = eval(out.split("\n")[0].split(":")[1].strip())
                    assert grade in [1,2,3,4,5]
                    return grade
                except:
                    pass
                print(f"Error in the generation of the grade")
                print( out)
                n_try-=1
        return grade
    
class OpenCodeInterpreter_2(OpenAI_Rank):
    def __init__(self, openai_key,puzzle_dict,mode_rank="absolute",prompt_instruction=None,model_id="gpt-3.5-turbo-0125",n_generation=1,temperature=0) -> None:
        self.prompt_2="""Rate the following code queries on a scale of 1 to 5 based on their complexity, where 1 is the easiest and 5 is the most
difficult. Consider the complexity of the query 
Query: [{query}]
You are obliged to choose only from the following list.
Scoring Criteria:
1 Point - Moderately Difficult: Involves understanding specific programming concepts or libraries, and may include
medium complexity algorithms or data structures like basic sorting algorithms or tree structures.
2 Points - Challenging: Requires handling more complex logic or algorithms such as advanced sorting algorithms,
recursive logic, or intermediate data structures like hash tables and heaps.
3 Points - Highly Challenging: Demands deeper knowledge in algorithms and data structures, potentially including
graph algorithms, dynamic programming, or complex string manipulation techniques.
4 Points - Advanced: Focuses on proficiency in programming and algorithm design, dealing with complex system
architecture issues, performance optimization, or solving advanced algorithmic challenges like NP-hard problems.
5 Points - Expert Level: The highest difficulty level, requiring innovative problem-solving approaches or unique
algorithm design, possibly involving interdisciplinary knowledge or the application of cutting-edge technologies.
Please give the score first with the format: "Score: [SCORE]" (write the score between bracket) then explain why"""
        super().__init__(openai_key=openai_key,puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,n_generation=n_generation,temperature=temperature)
        
    def absolute_grade(self,puzzle):
        """return the absolute_grade float between 0 and 10"""

        input_single = self.prompt_2.format(query=puzzle)
        n_try=3
        grade=-1
        while n_try>0:
            try:
                out = self.generate(input_single)
                grade=eval(out.split("[")[1].split("]")[0])
                assert grade in [1,2,3,4,5]
                return grade
            except:
                try:
                    grade = eval(out.split("\n")[0].split(":")[1].strip())
                    assert grade in [1,2,3,4,5]
                    return grade
                except:
                    pass
                print(f"Error in the generation of the grade")
                n_try-=1
                print( out)
        return grade

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
        # self.model=torch.compile(self.model)

    def generate(self,text):
        with torch.inference_mode():
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
    def __init__(self, puzzle_dict,mode_rank="pairwise",prompt_instruction=None,exllama2=True,model_id="GAIR/autoj-13b-GPTQ-4bits",n_generation=4) -> None:
        self.exllama2 = exllama2
        super().__init__(puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,n_generation=n_generation)
        
        
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



class Yes_model(HF_Rank):
    def __init__(self, puzzle_dict,mode_rank="absolute",prompt_instruction=None,exllama2=False,model_id="/home/flowers/work/hf/deepseek-coder-1.3b-instruct",yes_mode="finetuning",n_generation=1) -> None:
        """
        yes_mode = ["finetuning","education"] #prompt to use for the ranking
        """
        self.exllama2 = exllama2
        self.yes_mode = yes_mode # "finetuning" or "education"
        self.soft = torch.nn.Softmax(dim=1)
        super().__init__(puzzle_dict=puzzle_dict,mode_rank=mode_rank,prompt_instruction=prompt_instruction,model_id=model_id,exllama2=exllama2,n_generation=n_generation)
        
        
    def generate(self,text):
        with torch.inference_mode():
            time_s = time.time()
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            out_yes = self.model(**inputs)
            # out = self.tokenizer.decode(out_tok[0])
            k=10
            yes_logits=self.soft(out_yes.logits[:,-1]).cpu().detach() #logits associated with the token "yes"
            values,indices=torch.topk(yes_logits, k)
            list_token=self.tokenizer.batch_decode(indices.T)
            # values,list_token
            flag_no = False
            if "Yes" in list_token:
                idx = list_token.index("Yes")
                proba_Yes = values[[0],idx].item()
                if "yes" in list_token:
                    idx_yes = list_token.index("yes")
                    proba_yes = values[[0],idx_yes].item()
                    if proba_yes>proba_Yes:
                        idx = idx_yes
            elif "yes" in list_token:
                idx = list_token.index("yes")
            elif "No" in list_token:
                idx = list_token.index("No")
                flag_no = True
                proba_No = values[[0],idx].item()
                if "no" in list_token:
                    idx_no = list_token.index("no")
                    proba_no = values[[0],idx_no].item()
                    if proba_no>proba_No:
                        idx = idx_no
            elif "no" in list_token:
                idx = list_token.index("no")
                flag_no = True
            else:
                print("No yes or no token found")
                return -1
            proba_yes=values[[0],idx].item()
            if flag_no: # if the token "no" is selected, we need to invert the probability
                proba_yes = 1-proba_yes

            proba_yes=values[[0],idx].item()
            time_e = time.time()
            speed = inputs.input_ids.shape[-1]/((time_e-time_s)) # result in tokens per second
            self.list_speed_inference.append(speed)
            if self.speed_inference==None:
                self.speed_inference =  speed
                # compute ema speed inference
            else:
                alpha = 0.2
                self.speed_inference = self.speed_inference
                self.speed_inference = (speed * alpha) + (self.speed_inference * (1-alpha))
        return proba_yes
    
    def absolute_grade(self,puzzle):
        """return the absolute_grade float between 0 and 10"""
        # query = self.prompt_instruction

        input_single = build_Yes_input(datapoint=puzzle, 
                                model_id =self.model_id,yes_mode=self.yes_mode) # for single response evaluation 
        out = self.generate(input_single)
        return out



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

instruction_prometheus="""###Task Description:
An instruction, a response to evaluate and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Score Rubrics:
{Criteria}

###Feedback:
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