import json
import re
import time
import warnings
from typing import Optional, Union
import copy
import os
os.environ['TRANSFORMERS_CACHE'] = "models"
import numpy as np
import requests
# from openai.embeddings_utils import cosine_similarity, get_embedding
from openai import OpenAI
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def get_embedding(client, text, model="text-embedding-ada-002"): #self.config.embedding_model_path
#     return client.embeddings.create(input = [text], model=model).data[0].embedding


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer # maybe this is the pb (bitsandbytes launch message when doing multiprocess)?
import torch
# from langchain.chat_models import ChatOpenAI
from openelm.configs import P3ProblemEnvConfig, P3ProbSolEnvConfig, P3ProbSolChatEnv_PP_ELM_NLP_Config
from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.p3 import (
    P3_IMPORTS,
    P3_PROBLEM_LONG_SEED,
    P3_PROBLEM_MED_SEED,
    P3_PROBSOL_LONG_SEED,
    P3_PROBSOL_MED_SEED,
    create_prompt_label,get_class_PuzzleCheck,Topics_evaluation,skill_list
)
from openelm.environments.p3 import get_programming_puzzles_prompt,prompt_solve_puzzle_given_f,skills_evaluation,P3_probsol_chat_med_seed_goal_targeted
from openelm.mutation_model import MutationModel
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils.code_eval import pass_at_k, pool_exec_processes, type_check
from openelm.utils.code_eval import load_examples_p3,get_limited_trainset,just_remove_example_in_docstring,sample_target_skill_smart,sample_fewshot_example
# from joblib import parallel_config
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import transformers

# non-local imports, move quality in openelm?

from openelm.quality_metrics import utils
from openelm.quality_metrics.utils import load_prompt_PP
from openelm.quality_metrics.dataset_progress.progress_metrics import get_solution_logprobs


from tqdm import tqdm
class P3Solution(Genotype):
    def __init__(self, program_str: str, result_obj: dict, config: P3ProblemEnvConfig):
        """
        Genotype for a programming puzzle solution.
        Args:
            program_str: the solution program string (the g6() function).
            result_obj: dict.
            config: environment config
        """
        self.program_str = program_str
        self.result_obj = result_obj
        self.config = config

        # When comparing for phenotype, just use import statement and new solution function
        baseline = '''from typing import List

def g1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world")'''
    
        # self.baseline_emb = np.array(
        #     get_embedding(baseline, engine=self.config.embedding_model_path)
        # )

        if self.config.embedding_model_type == "hf":
            # when the model can't be loaded, with feat-extraction
            if self.config.embedding_model_path =="Salesforce/codet5p-110m-embedding":
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(self.config.embedding_model_path, trust_remote_code=True)
            self.pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )

    def to_phenotype(self) -> Optional[Phenotype]:
        if self.config.embedding_model_type == "openai":
            raise NotImplemented
            # compare_str = self.program_str
            # i_assert = compare_str.find("assert")
            # if i_assert > -1:
            #     compare_str = compare_str[:i_assert]
            # emb = np.array(
            #     get_embedding(compare_str, engine=self.config.embedding_model_path)
            # )
            # return cosine_similarity(emb, self.baseline_emb)
        elif self.config.embedding_model_type == "hf":
            if self.config.embedding_model_path =="Salesforce/codet5p-110m-embedding":
                with torch.no_grad():
                    inputs = self.tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt")
                    embedding = self.model(inputs)[0]
                    del self.tokenizer
                    del self.model
                return embedding
            else:
                features = np.array(self.pl(self.program_str))
                del self.pl
                return features.mean(axis=0).flatten()

    def __str__(self) -> str:
        return self.program_str

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pl" in state:
            del state["pl"]
        if "scaler" in state:
            del state["scaler"]
        if "pca" in state:
            del state["pca"]
        return state


class P3Problem(BaseEnvironment[P3Solution]):
    def __init__(
        self,
        config: P3ProblemEnvConfig,
        mutation_model: MutationModel,
        problem_str: str = None,
        solution_preamble: str = None,
    ) -> None:
        """
        The objective is to generate solutions to a given programming puzzle problem.
        Args:
            seed: the seed dict.
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            problem_str: an optional puzzle problem
            solution_preamble: accompanies optional problem_str
        """
        self.mutation_model = mutation_model
        self.config = config
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = None

        if self.config.prompt_size == "long":
            self.prompt_seed = P3_PROBLEM_LONG_SEED
        elif self.config.prompt_size == "med":
            self.prompt_seed = P3_PROBLEM_MED_SEED
        else:
            raise ValueError("No seed string found")

        # Get info for the puzzle that will be solved
        if problem_str is None:
            # This puzzle is at the index of the puzzles array specified by self.seed_index
            puzzles = requests.get(
                "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
            ).json()
            puzzle = puzzles[self.seed_index]

            self.problem_func = puzzle["sat"].replace(
                "def sat(", "def f6("
            )  # prompt form is f6()
            self.solution_preamble = puzzle["sol_header"].replace(
                "def sol(", "def g6("
            )  # solution form is g6()
            if self.config.prompt_size == "long":
                self.solution_preamble += (
                    "\n" + puzzle["sol_docstring"]
                )  # add in the docstring
            self.ans_type = puzzle["ans_type"]
        else:
            self.problem_func = problem_str
            self.solution_preamble = solution_preamble
            # TODO: generate a docstring?
            self.ans_type = None

        # Use the first example in the prompt seed as basis for embedding sizes
        i_first = self.prompt_seed.find("assert")
        first_example = self.prompt_seed[:i_first].strip()

        if self.config.embedding_model_type == "openai":
            self.genotype_ndim: int = 1
            self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
        elif self.config.embedding_model_type == "hf":
            # Dummy to get behavior space shape
            dummy_pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            dummy_scaler = StandardScaler()
            dummy_features = np.array(dummy_pl(first_example))
            dummy_features_scaled = dummy_scaler.fit_transform(
                np.squeeze(dummy_features)
            )
            dummy_pca = PCA(0.95)
            dummy_pca_features = dummy_pca.fit_transform(
                np.squeeze(dummy_features_scaled)
            )
            self.genotype_ndim: int = dummy_pca_features.shape[-1]
            self.genotype_space = np.repeat([[-20, 20]], self.genotype_ndim, axis=0).T

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str = self.prompt_seed

        prompt_str += f"\n\n{self.problem_func}"  # add this particular problem, f6(), to the prompt
        if code_batch is None:
            prompt_str += "\n"
        else:
            prompt_str += (
                "\n\n# Old version of g6()\n# TODO: fix bugs in the code below\n"
            )
            if isinstance(code_batch, list):
                # TODO: get nearby genotypes
                prompt_str += code_batch[0]
            elif isinstance(code_batch, str):
                prompt_str += code_batch

            prompt_str += "\n\n# Fixed version of g6()"

        prompt_str += f"\n{self.solution_preamble}"

        template = f"{P3_IMPORTS}\n{self.solution_preamble}"
        return {"prompt": prompt_str, "template": template}

    def generate_programs(self, code_batch: list[str]) -> list[P3Solution]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = True
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec, do_trunc=False
        )

        for i, gp in enumerate(generated_programs):
            i_assert = gp.find("assert")
            generated_programs[i] = gp[:i_assert].strip()

        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.sandbox_server}/eval_p3_solution",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            # TODO: handle (probably inside of pool_exec_processes) all cases
            # where the generated code returns a generator type. The multithreaded
            # execution pickles things and generators can't be pickled which
            # causes the whole thing to error out.
            # For now, try/except and re-try.
            try:
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g6",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except Exception:
                return self.generate_programs(code_batch)

        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config}
            for (gen_prog, res_obj) in zip(generated_programs, results)
        ]
        return [P3Solution(**p) for p in results]

    def evaluate_solution(self, sol: P3Solution) -> bool:
        """
        Returns whether or not the solution solves this problem
        """
        if self.ans_type is not None:
            return type_check(self.ans_type, sol.result_obj)

        eval_code = (
            f"{P3_IMPORTS}\n"
            f"{self.problem_func}\n"
            f"def run_eval():\n"
            f"    return f6({sol.result_obj})"
        )

        result = pool_exec_processes(
            eval_code,
            func_name="run_eval",
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        return result[0]

    def fitness(self, sol: P3Solution) -> float:
        """
        If passing the solution to the problem returns True, fitness is 1.0
            else -np.inf
        """
        result = self.evaluate_solution(sol)

        if result is True:
            return 1.0
        else:
            return -np.inf

    def random(self) -> list[P3Solution]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_solutions = self.generate_programs(program_list)
        return new_solutions

    def mutate(self, sol_list: list[P3Solution]) -> list[P3Solution]:
        sols = [s.program_str for s in sol_list]
        program_list = list(map(self.construct_prompt, sols))
        new_sols = self.generate_programs(program_list)
        return new_sols

# the one to use with P3ProbSol_Chat
class P3ProbSolResult(Genotype):
    def __init__(self, program_str: str, config: P3ProbSolEnvConfig, emb: list= None,
                  idx_generation: int=-1,target_skills=None,fitness: int =None, quality: int =None,
                  description:str="[description of the puzzle]", interestingness_f:int=None, interestingness_g:float=None, is_valid:bool=None, is_valid_explanation:str=None,result_obj : Optional[dict]={}, explanation_emb=None) -> None:
        """
        Genotype for a programming puzzle problem+solution pair.
        Args:
            program_str: the code for the pair.
            result_obj: the result of the solution.
            config: environment config
        """
        self.fitness=fitness
        self.program_str = program_str
        self.result_obj = result_obj
        self.config = config
        self.emb = emb
        self.explanation_emb = explanation_emb
        self.idx_generation = idx_generation
        self.target_skills = target_skills
        if self.config.env_name == "p3_probsol_Chat" :
            # print("not implemented yet")
            # i_f = program_str.find("def f(")
            i_g = program_str.find("def g(")
            
            self.problem_func = self.program_str[:i_g].strip()
            self.solution_func = self.program_str[i_g:].strip()
            # no more problem if an assert is in def f
            i_assert = self.solution_func.find("assert") 
            self.solution_func = self.solution_func[:i_assert].strip() 
        
        self.quality = quality,
        self.description=description,
        self.is_valid = is_valid, 
        self.is_valid_explanation = is_valid_explanation

        if self.config.GPT_feedback:
            self.interestingness_f = interestingness_f, 
            self.interestingness_g = interestingness_g,




    def __str__(self) -> str:
        return self.program_str
    
    def __to_dict__(self) -> dict:
        if self.emb is None:
            self.emb = []
        if self.target_skills is None:
            self.target_skills = []
        dic={"fitness":self.fitness,"program_str":self.program_str, "emb":list(self.emb),"explanation_emb":self.explanation_emb}
        dic.update({"idx_generation":self.idx_generation,"target_skills":list(self.target_skills)})
        dic.update({"quality":self.quality,"description" : self.description, "interestingness_f": self.interestingness_f})
        dic.update({"interestingness_g":self.interestingness_g,"is_valid":self.is_valid,"is_valid_explanation":self.is_valid_explanation})
        return dic

    def to_phenotype(self) -> Optional[Phenotype]:
        if not self.emb is None:
            return self.emb
        else: 
            if self.config.embedding_model_type == "openai":
                # Openai backend to get the embedding
                if "embedding" in self.config.embedding_model_type: 
                    raise NotImplemented
                    # use the embedding model to get the embedding
                    # compare_str = (
                    #     self.program_str
                    # )  # TODO: remove comments from f6_2 for diversity measurement
                    # i_assert = compare_str.find("assert f")
                    # if i_assert > -1:
                    #     compare_str = compare_str[:i_assert]
                    # emb = np.array(
                    #     get_embedding(compare_str, engine=self.config.embedding_model_path)
                    # )
                    # return emb
                else: 
                    #use GPT to get the "embedding" in NLP space
                    raise "can't do that in the Genotype class, should be done in the P3 environment"
            
            elif self.config.env_name == "p3_probsol_Chat" and self.config.embedding_model_type == "hf": 
                # Huggingface backend to get the embedding
                # when the model can't be loaded, with feat-extraction
                if self.config.embedding_model_path =="Salesforce/codet5p-110m-embedding":
                    tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model_path, trust_remote_code=True)
                    model = AutoModel.from_pretrained(self.config.embedding_model_path, trust_remote_code=True)
                    with torch.no_grad():
                        inputs = tokenizer.encode(self.program_str, return_tensors="pt")
                        embedding = model(inputs)[0]
                        del tokenizer
                        del model
                    return embedding.numpy()#.tolist()
                else:
                    pl = pipeline(
                        "feature-extraction", model=self.config.embedding_model_path
                    )
                    features = np.array(pl(self.program_str))
                    del pl
                    return features.mean(axis=0).flatten()

            else:
                raise NotImplementedError

    def __getstate__(self):
        state = self.__dict__.copy()
        if "pl" in state:
            del state["pl"]
        if "scaler" in state:
            del state["scaler"]
        if "pca" in state:
            del state["pca"]
        return state


class P3ProbSol(BaseEnvironment[P3ProbSolResult]):
    def __init__(
        self,
        config: P3ProbSolEnvConfig,
        mutation_model: MutationModel,
    ) -> None:
        """
        The objective is to generate problem+solution pairs.
        Args:
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            ans_type: answer type
        """
        self.mutation_model = mutation_model
        self.config = config
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = None

        if self.config.prompt_size == "long":
            self.prompt_seed = P3_PROBSOL_LONG_SEED
        elif self.config.prompt_size == "med":
            self.prompt_seed = P3_PROBSOL_MED_SEED
        else:
            raise ValueError("No seed string found")

        # Use the first example in the prompt seed as basis for embedding sizes
        i_first = self.prompt_seed.find("assert")
        first_example = self.prompt_seed[:i_first].strip()

        if self.config.embedding_model_type == "openai":
            raise NotImplemented
            # dummy_features = np.array(
            # get_embedding(first_example, engine=self.config.embedding_model_path))
            # self.genotype_ndim: int = len(dummy_features)
            # self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
        elif self.config.embedding_model_type == "hf":
            # Dummy to get behavior space shape
            dummy_pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
            dummy_features = np.array(dummy_pl(first_example))
            dummy_scaler = StandardScaler()
            dummy_features_scaled = dummy_scaler.fit_transform(
                np.squeeze(dummy_features)
            )
            dummy_pca = PCA(0.95)
            dummy_pca_features = dummy_pca.fit_transform(dummy_features_scaled)
            self.genotype_ndim: int = dummy_pca_features.shape[-1]
            self.genotype_space = np.repeat([[-20, 20]], self.genotype_ndim, axis=0).T

        # Get info for the seed puzzle that will be mutated
        # This puzzle is at the index of the puzzles array specified by self.seed_index
        # TODO: put this in a method or in construct_prompt()?
        puzzles = requests.get(
            "https://raw.githubusercontent.com/microsoft/PythonProgrammingPuzzles/v0.2/puzzles/puzzles.json"
        ).json()
        puzzle = puzzles[self.seed_index]
        if len(puzzle["sol_bodies"]) == 0:
            raise ValueError(
                f"No sample solution is provided for the puzzle at index {self.seed_index}"
            )

        f6_1 = puzzle["sat"].replace("def sat(", "def f6_1(")  # problem form is f6_1()
        g6_1 = puzzle["sol_header"].replace(
            "def sol(", "def g6_1("
        )  # solution form is g6_1()
        if self.config.prompt_size == "long":
            g6_1 += "\n" + puzzle["sol_docstring"]  # add in the docstring
        g6_1 += (
            "\n" + puzzle["sol_bodies"][0]
        )  # include the first example solution function body

        self.original_probsol = f6_1 + "\n\n" + g6_1 + "\n\n" + "assert f6_1(g6_1())"
        self.new_probsol_preamble = "def f6_2("
        self.preprocess_p3()
        
    def preprocess_p3(self):
        trainset = load_examples_p3()
        if self.config.limited_trainset:
            trainset=get_limited_trainset()
        list_p3 = [P3ProbSolResult(**p) for p in trainset]

        self.archive_P3puzzle = list_p3

            
                

        
    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        prompt_str = self.prompt_seed

        if code_batch is None:
            # prompt with prob+sol from P3 dataset
            prompt_str += (
                f"\n\n{self.original_probsol}"  # add this particular probsol, f6_1() and g6_1(), to the prompt
                f"\n\n{self.new_probsol_preamble}"  # add f6_2() preamble to the prompt
            )
        else:
            # prompt with prob+sol that is given (one that was the output of a prev mutation)
            if isinstance(code_batch, list):
                # TODO: get nearby genotypes
                program_str = code_batch[0]
            elif isinstance(code_batch, str):
                program_str = code_batch

            # the prev output was f6_2 and g6_2, so now make it f6_1 and g6_1 for the prompt
            # and remove comments (which contain changes from prev f6_1) from new f6_1
            # TODO: pass in the whole object instead of the program_str since it already parsed some of this?
            i_f6 = program_str.find("def f6_2")
            program_str = program_str[i_f6:]  # remove import statement
            program_str = program_str.replace("f6_2(", "f6_1(")
            program_str = program_str.replace("g6_2(", "g6_1(")
            i_g6 = program_str.find("def g6_1(")
            # remove comments with """
            program_str = (
                re.sub('""".*"""', "", program_str[:i_g6]) + program_str[i_g6:]
            )
            # remove comments with # (and remove empty lines)
            i_g6 = program_str.find("def g6_1(")
            lines = program_str[:i_g6].strip().split("\n")
            new_lines = []
            for line in lines:
                if line.strip().startswith("#") or len(line.strip()) == 0:
                    continue
                new_lines.append(line)
            program_str = "\n".join(new_lines) + "\n\n" + program_str[i_g6:]
            program_str = program_str.strip()

            prompt_str += f"\n\n{program_str}" f"\n\n{self.new_probsol_preamble}"

        template = f"{P3_IMPORTS}\n{self.new_probsol_preamble}"
        return {"prompt": prompt_str, "template": template}

    def generate_programs(self, code_batch: list[str]) -> list[P3ProbSolResult]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = False
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec
        )

        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.sandbox_server}/eval_p3_solution",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
        else:
            # TODO: handle (probably inside of pool_exec_processes) all cases where the generated code returns
            # a generator type. The multithreaded execution pickles things and generators can't be pickled
            # which causes the whole thing to error out.
            # For now, try/except and re-try.
            
            try:
                generated_programs = [code.split("assert")[0] for code in generated_programs]
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g6_2",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except Exception:
                return self.generate_programs(code_batch)

        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config}
            for (gen_prog, res_obj) in zip(generated_programs, results)
        ]
        return [P3ProbSolResult(**p) for p in results]

    def fitness(self, probsol: P3ProbSolResult) -> float:
        """
        Fitness is the inverse of pass@k of the problem func.
        We want a pass@k of >0 so that the problem is reasonably solvable.
        So fitness=0 if unsolved (which is still better than -np.inf).
        Other than that, more difficult (lower pass@k) => higher fitness.
        """
        if isinstance(probsol.result_obj, ExecResult):
            return -np.inf

        # TODO: check type expected by f6_2 if any?
        # TODO: implement checks for absolute triviality of f6_2 requirements
        #   the fitness function being based on pass@k might take care of this though

        eval_code = (
            f"{P3_IMPORTS}\n"
            f"{probsol.problem_func}\n"
            f"def run_eval():\n"
            f"    return f6_2({probsol.result_obj})"
        )
        

        # Run code to see if g6_2 solves f6_2
        result = pool_exec_processes(
            eval_code,
            func_name="run_eval",
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        # if result[0] is True: what  result[0]== True is the problem is solved
            # return -np.inf
        
        # if just one try more like
        if result[0] is True and self.config.eval_k <= 1:
            return 1.0
        
        
        # Do pass@k eval

        # Get f6_2() and make it the new f6()
        problem_str = probsol.problem_func.replace("def f6_2(", "def f6(")
        # Remove comments with """
        problem_str = re.sub('""".*"""', "", problem_str)
        # Remove comments with # (and remove empty lines)
        lines = problem_str.strip().split("\n")
        new_lines = []
        for line in lines:
            if line.strip().startswith("#") or len(line.strip()) == 0:
                continue
            new_lines.append(line)
        problem_str = "\n".join(new_lines)
        # Get solution_preamble for g6()
        i_end_preamble = probsol.solution_func.find("):")
        solution_preamble = probsol.solution_func[: i_end_preamble + 2].replace(
            "def g6_2(", "def g6("
        )

        p3_problem = P3Problem(
            self.config,  # TODO: make an actual P3ProblemEnvConfig
            self.mutation_model,
            problem_str=problem_str,
            solution_preamble=solution_preamble,
        )
        solutions = []
        for _ in range(self.config.eval_k // self.config.batch_size + 1):
            solutions += p3_problem.random()

        c = 0
        for s in solutions:
            if p3_problem.evaluate_solution(s) is True:
                c += 1

        pak = pass_at_k(len(solutions), c, self.config.eval_k)
        return 1 / pak if pak > 0 else 0

    def random(self) -> list[P3ProbSolResult]:
        # need to multiprocess that
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)] 
        new_probsols = self.generate_programs(program_list)
        return new_probsols

    def mutate(self, probsol_list: list[P3ProbSolResult]) -> list[P3ProbSolResult]:
        probsols = [pb.program_str for pb in probsol_list]
        program_list = list(map(self.construct_prompt, probsols))
        new_probsols = self.generate_programs(program_list)
        return new_probsols




# chatGPT version of Probsol


class P3ProbSol_Chat(BaseEnvironment[P3ProbSolResult]):
    def __init__(
        self,
        config: P3ProbSolEnvConfig,
        mutation_model: MutationModel,
    ) -> None:
        """
        /!\ optimized for chatGPT /!\
        compare to base prob sol:
        remove the explicit mutation in the prompt (prompt with underscore i_1 i_2) as it guided to much the model
        and it lead to bad diversity of generated problems.
        
        The objective is to generate problem+solution pairs.
        Args:
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            ans_type: answer type
        """
        self.n_skills = len(skill_list)
        self.mutation_model = mutation_model
        self.config = config
        print(f" \n\n ======================\n\n ======================\n\n{self.config.IMGEP_mode} \n\n ======================\n\n\n ======================\n\n")
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = np.random.default_rng(self.config.seed)

        if self.config.prompt_size == "long":
            raise ValueError("long prompt no implemented yet ")
        elif self.config.prompt_size == "med":
            self.prompt_seed_function = get_programming_puzzles_prompt
            # self.prompt_seed= self.prompt_seed_function()
        else:
            raise ValueError("No seed string found")


        
        #load embedding model for the phenotype
        print("load embedding model:" )
        print(self.config.embedding_model_path)
        if self.config.embedding_model_type == "hf": 
            # when the model can't be loaded, with feat-extraction
            if self.config.embedding_model_path =="Salesforce/codet5p-110m-embedding":
                print( "mode tokenzier + model from huggingface hub")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(self.config.embedding_model_path, trust_remote_code=True)
            else:
                print( "mode pipeline from huggingface hub")
                self.pl = pipeline(
                "feature-extraction", model=self.config.embedding_model_path
            )
        
        first_example ="def f(x,a=1,b=1): return a*x==b \ndef g(x,a=1,b=1): return b/a\nassert f(g())==True"
        # self.n_skills = n_skills
        out = self.to_phenotype(first_example)["emb"]
        if self.config.embedding_model_type == "openai" and not "embedding" in self.config.embedding_model_type: 
            #NLP space

            self.genotype_ndim = np.array(out).shape[-1]
            #  in poetry self.genotype_space = np.array(self.config.behavior_space).T
            self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T 
        else:
            self.genotype_ndim = np.array(out).shape[-1]
            self.genotype_space = np.repeat([[-1, 1]], self.genotype_ndim, axis=0).T
        
        if self.config.use_preprocessed_trainset:
            # preprocessing of the trainset
            print("loading preprocessed trainset")
            self.preprocess_p3()
                
            

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass


    def label_puzzle(self,program_str) -> dict:
        """
        Label a puzzle with the skills it requires
        TODO: add a safeguard if the model hallucinate too much e.g len(category_idx_predicted) > n_skills
        """
        
        prompt = create_prompt_label(program_str)
        tool_skill_labeling = Topics_evaluation
        result=self.mutation_model.generate_completion_instructor(list_prompt = [prompt],batch_tools=[tool_skill_labeling],temperature=0.,activate_parrallel=False)[0]
        skill=result.index_topics
        explanation_skill =result.explanations_index_topics
        if not len(skill)<=5: # we should have at most 5 topics
            skill=skill[:5]
        skill =[1 if i in skill else 0 for i in range(self.n_skills)]
        # tool_diversity = Puzzle_Diversity
        # tool_interstingness = Puzzle_Interestingness
        dic_label={"emb":skill,"explanation_emb":explanation_skill}
        return dic_label

    
    def to_phenotype(self,program_str: str):
        """compute embedding of the program"""
        # "regular" embedding
        if self.config.GPT_feedback: 
            #use chatGPT (or GPT model) to get the "embedding" in NLP space
            return self.label_puzzle(program_str)
        
        elif self.config.embedding_model_type == "openai":
            raise NotImplemented
            # if "embedding" in self.config.embedding_model_type: 
            #     emb = np.array(
            #         get_embedding(program_str, engine=self.config.embedding_model_path))
            #     return emb
    
        elif self.config.embedding_model_type == "hf": 
            # when the model can't be loaded, with feat-extraction
            if self.config.embedding_model_path =="Salesforce/codet5p-110m-embedding":
                with torch.no_grad():
                    inputs = self.tokenizer.encode(program_str, return_tensors="pt",truncation=True,max_length=512)
                    emb = self.model(inputs)[0]
                return {"emb":emb.numpy()}
            
            elif self.config.embedding_model_type == "hf":
                # weird preprocessing 
                features = np.array(self.pl(program_str))

                return {"emb":features.mean(axis=0).flatten()} # mean pooling
            
        else:
            raise NotImplementedError


    def to_multiple_phenotype(self, list_program_str: List[str]):
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]
        if self.config.GPT_feedback:
            # for api based model
            completions=[]
            max_workers = self.mutation_model.config.processes
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for sub_batch in chunks(list_program_str, max_workers):
                    for idx,message_list in enumerate(sub_batch):
                        kwargs={"program_str":message_list}
                        # if "kwargs" in kwargs_modified:
                        #     original_kwargs = kwargs_modified.pop("kwargs")
                        future = executor.submit(
                            self.to_phenotype,**kwargs
                        )
                        completions.append(future)
            # Retrieve the results from the futures
            list_phenotype_correct_puzzle = [future.result() for future in completions]
        else:
            # for local model
            list_phenotype_correct_puzzle = []
            for program_str in list_program_str:
                list_phenotype_correct_puzzle.append(self.to_phenotype(program_str))
        return list_phenotype_correct_puzzle

    def description_filtering(self,program_str):
        """give a description of the puzzle and a boolean if it is pass the filter or not"""
        if self.config.activate_filtering_description and self.config.puzzle_filtering:
            mode = "description+is_valid"
        elif self.config.activate_filtering_description and not self.config.puzzle_filtering:
            mode = "description"
        else: NotImplementedError("should not go there")

        prompt = create_prompt_label(program_str, mode = mode)
        tool_skill_labeling = get_class_PuzzleCheck(mode)

        result=self.mutation_model.generate_completion_instructor(list_prompt = [prompt],batch_tools=[tool_skill_labeling],temperature=0.,activate_parrallel=False)[0]
        dic_features = {}
        if "description" in mode: 
            puzzle_description = result.puzzle_description
            dic_features["description"] = puzzle_description
        if "is_valid" in mode:
            is_valid_explanation = result.explanations
            is_valid = result.give_puzzle_to_student
            dic_features.update({"is_valid_explanation": is_valid_explanation, "is_valid": is_valid})
        return dic_features

    def multiple_description_filtering(self,list_program_str: List[str]):
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]
        # for api based model
        completions=[]
        max_workers = self.mutation_model.config.processes
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch in chunks(list_program_str, max_workers):
                for idx,message_list in enumerate(sub_batch):
                    kwargs={"program_str":message_list}
                    # if "kwargs" in kwargs_modified:
                    #     original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(
                        self.description_filtering,**kwargs
                    )
                    completions.append(future)
        # Retrieve the results from the futures
        list_description_filter = [future.result() for future in completions]
        return list_description_filter

    def preprocess_p3(self, split="train",load_embedding=True,debug=False):
        """
        Preprocess the trainset of P3 
        load embedding from json files 
        debug give random embedding to the puzzles for debugging purpose
        """
        load_embedding = self.config.use_preprocessed_trainset_emb
        if self.config.limited_trainset:
            trainset=get_limited_trainset()
        else:
            trainset = load_examples_p3()
        # print("start loading p3 trainset into map")
        # trainset = preprocessing_P3(split =split, n_token_max=512,load_embedding = load_embedding,debug=debug)
        
        for puz in tqdm(trainset):
            puz["config"] = self.config
            
            if not load_embedding or self.config.embedding_model_type == "hf":
                puz["emb"]=self.to_phenotype(puz["program_str"])["emb"]
                
        #     puz["program_str"] = just_remove_example_in_docstring(puz["program_str"]) # remove ex in docstring       
        list_p3 = [P3ProbSolResult(**p) for p in trainset]

        self.archive_P3puzzle = list_p3



    def mutate_vec(self, vec2mutate,k=1):
        """
        take an vector and mutate k values randomly (from 0. to 1. or 1. to 0.)
        """
        vec_mutate=copy.deepcopy(vec2mutate)
        idx = self.rng.choice(vec_mutate.shape[0], k, replace=False)
        vec_mutate[idx] = 1.-vec_mutate[idx]
        return vec_mutate
    
    def construct_prompt(
        self, list_phenotype,skill_targeted=[], trainset_only = False
    ) -> dict[str, str]:
        """
        construct the prompt for the LLM
        """

        if not isinstance(skill_targeted, list):
            skill_targeted=skill_targeted.tolist()

        if trainset_only: # rd gen
            prompt_str = self.prompt_seed_function(list_few_shot_example=list_phenotype)                    
        
        elif skill_targeted == []: # elm mode
            # code_batch is puzzle to mutate (last puzzle of list_phenotype)
            code_batch = list_phenotype[-1]
            list_phenotype = list_phenotype[:-1]
            prompt_str = self.prompt_seed_function(list_few_shot_example=list_phenotype, code_batch=code_batch)
        else:
            prompt_str = self.prompt_seed_function(list_few_shot_example=list_phenotype,skill_targeted=skill_targeted)

        template = f"{P3_IMPORTS}\n"#{self.new_probsol_preamble}"
        return {"prompt": prompt_str, "template": template},skill_targeted


    def generate_programs(self, code_batch: list[dict[str, str]],skill_targeted_list: list[Union[None,list[int]]]) -> list[P3ProbSolResult]:
        """Generate new programs with a mutation model parse them, compute fitness and evaluate them."""
        
        print('generating programs')
        local_scope_exec = False
        start_t0 = time.time()
        _generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec,do_trunc=False
        )
        print('done')
        start_t1 = time.time()
        
        list_pb=[]

        # parse the generated code
        # as we generate multiple puzzle for each batch_size (in total 3*batch_size) we need to associate skill targeted with the list of all pb (list_pb)

        skill_targeted_list_duplicate=[]
        for idx_gen_prog,gen_prog in enumerate(_generated_programs): #_generated_programs => batch_size params
            # should probably use regex (faster)
            split_pb = copy.deepcopy(gen_prog.replace("```python","```").replace("``` python","```").split("```"))# replace("```\n","```").
            for idx in range(len(split_pb)):
                if "def f" in split_pb[idx] and "def g" in split_pb[idx]:
                    list_pb.append(split_pb[idx])
                    skill_targeted_list_duplicate.append(skill_targeted_list[idx_gen_prog])

        for idx_assert in range(len(list_pb)):
        #     list_pb[idx] = list_pb[idx].split("assert")[0]+"assert f(g()) == True"
            if not "assert f(" in list_pb[idx_assert]:
                list_pb[idx_assert] = list_pb[idx_assert] + "\nassert f(g()) == True"
        generated_programs = list_pb
        print('parsing finished')
        print(f"time to generate {len(generated_programs)} program = {start_t1-start_t0} sec")
        
        print('evaluating pbs')
        list_lib = ["math", "random", "itertools"]
        
        for idx in range(len(generated_programs)):
            if "List" in generated_programs[idx] and not("from typing import List" in generated_programs[idx]):
                generated_programs[idx] = "from typing import List\n" + generated_programs[idx]
                
            # check if lib are correctly imported (if not import them)
            # for lib in list_lib:
            #     if lib in generated_programs[idx]:
            #         import_pb = generated_programs[idx].split("def f")[0]
            #         if not(f"import {lib}" in import_pb or f"from {lib}" in  import_pb):
            #             generated_programs[idx] = f"import {lib}\n" + generated_programs[idx]
    
        if self.config.sandbox:
            results = []
            for code in generated_programs:
                resp = requests.post(
                    f"{self.sandbox_server}/eval_p3_solution",
                    json={"code": code, "timeout": self.config.timeout},
                    timeout=self.config.timeout,
                )
                if resp.status_code == 200:
                    return_dict = json.loads(resp.text)
                    results.append(return_dict)
            print('done')

        # Just label correct problem to save computation time or $$ (chatGPT):
        pre_results = [
            {"program_str": gen_prog, "config": self.config, "idx_generation": self.idx_generation, "target_skills":target_skills,"fitness":-np.inf}
            for (gen_prog, target_skills) in zip(generated_programs, skill_targeted_list_duplicate)
        ]
        probsol_2_test = [P3ProbSolResult(**p) for p in pre_results]
        start_t4 = time.time()

        #compute fitness
        list_fitness = self.multiple_fitness(probsol_2_test) #[self.fitness(puzz) for puzz in probsol_2_test]
        start_t5 = time.time()
        print( f"time to compute {len(generated_programs)} fitness = {start_t5-start_t4}")
        idx_correct_puzzle = [idx for idx,fit in enumerate(list_fitness) if fit is -np.inf]#>= 0.0] # remove puzzle with fit<0 or just fit == -np.inf ?
        print(f"number of correct puzzle {len(idx_correct_puzzle)}")
        list_correct_puzzle = [generated_programs[idx] for idx in idx_correct_puzzle]

        # add gen description puzzle here (or should we do it with skill labeling ?) 
        # + add filtering step here ?
        if self.config.activate_filtering_description:
            add_to_results =self.multiple_description_filtering(list_correct_puzzle)
        

        # compute phenotype of correct puzzle
        start_t6 = time.time()
        
        print('begin phenotype computation')

        #TODO: use puzzle description in phenotype computation (add description to list_correct_puzzle)
        # if self.config.activate_filtering_description:
        #     for idx in range(len(list_correct_puzzle)):
        #         list_correct_puzzle[idx] = add_to_results[idx]["description"] +"\n"+ list_correct_puzzle[idx]
        
        list_phenotype_correct_puzzle = self.to_multiple_phenotype(list_correct_puzzle) # should probably give description to label puzzle ?
        # with parallel_config(n_jobs=self.config.processes, prefer="threads"): #backend='threading',
        #     list_phenotype_correct_puzzle = Parallel()(delayed(self.to_phenotype)(puzzl) for puzzl in list_correct_puzzle) # need to handle batch within self.to_phenotype
        start_t7 = time.time()
        print( f"time to compute phenotype for {len(list_correct_puzzle)} correct problem  = {start_t7-start_t6}")
        list_phenotype = [[-1] for _ in range(len(generated_programs))] # [-1] when eval is not correct

        # add phenotype of correct puzzle to the list of phenotype            
        generated_programs = [gen_prog for gen_prog in generated_programs]
        results = [
            {"program_str": gen_prog, "config": self.config, "emb": pheno, "idx_generation": self.idx_generation, "target_skills":target_skills,"fitness":fitness}
            for (gen_prog, target_skills,pheno,fitness) in zip(generated_programs, skill_targeted_list_duplicate,list_phenotype,list_fitness)
        ]

        # add embedding + description embedding to the results
        for idx,idx_puzzle in enumerate(idx_correct_puzzle):
            results[idx_puzzle].update(list_phenotype_correct_puzzle[idx])


        if self.config.activate_filtering_description: # add description and/or filtering to results
            for idx,idx_puzzle in enumerate(idx_correct_puzzle):
                results[idx_puzzle].update(add_to_results[idx])
            
        print('finished generation')
        return [P3ProbSolResult(**p) for p in results]
    
    
    def try_solving_problem(self, probsol: P3ProbSolResult) -> list[P3ProbSolResult]:
        """
        generate new solution to a problem given multiple time (can be used for computing pass@k)
        """
        new_probsol = copy.deepcopy(probsol)
        prompt= prompt_solve_puzzle_given_f(new_probsol.program_str)
        template = ""
        code_batch=[{"prompt":prompt,"template":template} for _ in range(self.config.eval_k-1)] # -1 because we already have the original problem
        local_scope_exec = False
        _generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec,do_trunc=False
        )
        
        # should we just ask LLM to correct g() or to correct the whole puzzle?
        list_pb=[]
        # parse the generated code 
        for gen_prog in _generated_programs:
            split_pb = copy.deepcopy(gen_prog.replace("```python","```").replace("```\n","```").split("```"))
            for idx in range(len(split_pb)):
                if "def g" in split_pb[idx] and "return " in split_pb[idx]:
                    list_pb.append(split_pb[idx])
        for idx_assert in range(len(list_pb)):
            if not "assert f(" in list_pb[idx_assert]:
                list_pb[idx_assert] = list_pb[idx_assert] + "\nassert f(g()) == True"
        generated_programs = list_pb
        
        list_new_puzzles = [probsol]
        for idx in range(len(generated_programs)):
            new_pb_str = new_probsol.program_str.split("def g(")[0] + generated_programs[idx]
            
            if "List" in generated_programs[idx] and not("from typing import List" in generated_programs[idx]):
                new_pb_str = "from typing import List\n"  + new_pb_str
                probsol_2_add=copy.deepcopy(new_probsol)
                probsol_2_add.program_str = new_pb_str
                list_new_puzzles.append(probsol_2_add)
        return list_new_puzzles       

    def fitness(self, probsol: P3ProbSolResult, use_pass_k = False) -> float:
        """
        Fitness is the inverse of pass@k of the problem func.
        We want a pass@k of >0 so that the problem is reasonably solvable.
        So fitness=0 if unsolved (which is still better than -np.inf).
        Other than that, more difficult (lower pass@k) => higher fitness.
        """
        # if isinstance(probsol.result_obj, ExecResult):
        #     return -np.inf

        # TODO pass@k eval
        if not probsol.fitness == None:
            return probsol.fitness

        prog = probsol.program_str.split("\nassert f")
        probsol.program_str = prog[0] + "\nassert f(g()) == True\n"
        eval_code_ = str(
            f"{probsol.program_str}\n"
            f"def run_eval():\n"
            f"    return f(g())"
        )
        eval_codes =[eval_code_]#1, eval_code_2]
        # Run code to see if g6_2 solves f6_2
        try:
            result = pool_exec_processes(
                eval_codes,
                func_name="run_eval",
                timeout=self.config.timeout,
                processes=self.config.processes,
                debug=self.config.debug,
            )
        except:
            result = [False]

        if self.config.eval_k<=1 : # one try doesn't compute pass@k
            if result[0] == True:
                return 1.0
            
            else:
                return -np.inf
            
        # compute pass@k
        else: # TODO; check if it is working
            list_new_puzzles = self.try_solving_problem(probsol)
        
            c = 0
            for idx_sol in range(len(list_new_puzzles)):
                p3probsol = list_new_puzzles[idx_sol]
                if self.fitness(p3probsol, use_pass_k = False) == 1.0:
                    
                    probsol.program_str = p3probsol.program_str
                    c+=1

            pak = pass_at_k(len(list_new_puzzles), c, self.config.eval_k)
            return 1 / pak if pak > 0 else 0


    def multiple_fitness(self,list_probsol: list[P3ProbSolResult], use_pass_k = False, parrallel_fitness=True):

        if parrallel_fitness:
            list_fitness = []
            eval_codes = []
            indices = []  # To keep track of the indices of the probsols being processed
            for index, probsol in enumerate(list_probsol):
                if probsol.fitness == -np.inf:
                    prog = probsol.program_str.split("\nassert f")
                    probsol.program_str = prog[0] + "\nassert f(g()) == True\n"
                    eval_code_ = str(
                        f"{probsol.program_str}\n"
                        f"def run_eval():\n"
                        f"    return f(g())"
                    )
                    eval_codes.append(eval_code_)
                    indices.append(index)

            # for probsol in list_probsol:
            #     prog = probsol.program_str.split("\nassert f")
            #     probsol.program_str = prog[0] + "\nassert f(g()) == True\n"
            #     eval_code_ = str(
            #         f"{probsol.program_str}\n"
            #         f"def run_eval():\n"
            #         f"    return f(g())"
            #     )
            #     eval_codes.append(eval_code_)
            # Run code to see if g6_2 solves f6_2
            try:
                partial_results = pool_exec_processes(
                    eval_codes,
                    func_name="run_eval",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except:
                partial_results = [False]*len(eval_codes) 
                print("pb when computing fitness")


            # Map the partial results back to the full list
            results = [puz.fitness for puz in list_probsol] # Initialize all fitness values with -np.inf
            for index, result in zip(indices, partial_results):
                if result:
                    results[index] = 1.0  # Update only those indices which were processed
            for idx in range(len(list_probsol)):
                list_probsol[idx].fitness = results[idx]
            assert len(list_probsol) == len(results), "pb when computing fitness"
            list_fitness = results
            # if self.config.eval_k<=1 : # one try doesn't compute pass@k

        else:
            list_fitness = [self.fitness(puzz) for puzz in list_probsol]
            for idx in range(len(list_probsol)):
                list_probsol[idx].fitness = list_fitness[idx]
            
        return list_fitness

    def random(self,batch) -> list[P3ProbSolResult]:
        # should just take few shot example from trainset not archive

        assert len(batch) == self.config.batch_size
        program_list = []
        skill_targeted_list = []
        for (list_few_shot_example_phenotypes, skill_targeted) in batch:
            dic_prompt, skill_targeted = self.construct_prompt(list_few_shot_example_phenotypes, skill_targeted, trainset_only=True)
            program_list.append(dic_prompt)
            skill_targeted_list.append(skill_targeted)

        new_probsols = self.generate_programs(program_list,skill_targeted_list)
        return new_probsols

    def mutate(self, batch: list) -> list[P3ProbSolResult]:
        # (list_few_shot_example_phenotypes, skill_targeted) = batch
        assert len(batch) == self.config.batch_size
        program_list = []
        skill_targeted_list = []
        for (list_few_shot_example_phenotypes, skill_targeted) in batch:
            dic_prompt, skill_targeted = self.construct_prompt(list_few_shot_example_phenotypes, skill_targeted, trainset_only=False)
            program_list.append(dic_prompt)
            skill_targeted_list.append(skill_targeted)


        new_probsols = self.generate_programs(program_list,skill_targeted_list)

        return new_probsols


class P3ProbSol_Chat_PP(P3ProbSol_Chat):
    def __init__(
        self,
        config: P3ProbSolChatEnv_PP_ELM_NLP_Config,
        mutation_model: MutationModel,
    ) -> None:
        """
        Version of the P3 Problem-solution environment with the Prediction Progress (PP)
        fitness measure. This emvironment implements the in=context version of PP.

        PP needs a model and an archive probsol dataset. For the currently evaluated
        probsol, we put it as an example in the prompt before a probsol from the archive
        and we measure how much loss decreases compared to a reference probsol. We average
        out this value for probsols on the whole archive.
        """
        self.archive_dataset_name = config.archive_dataset_name
        self.reference_probsol = config.reference_probsol
        one_shot_prompt_id = config.one_shot_prompt_id
        self.use_docstring = config.use_docstring
        # for computing the solution attention mask in parallel
        self.num_workers = config.num_workers
        self.batch_size = config.batch_size
        self.compile = config.compile
        self.flash_attn = config.flash_attn
        self.num_max_tokens = config.num_max_tokens

        super().__init__(config, mutation_model)

        # from vllm import LLM, SamplingParams
        # llm = LLM('microsoft/phi-1')

        # load model and tokenizer
        self.model, self.tokenizer = utils.create_model_and_tokenizer(
            config.model_or_model_path, compile=self.compile, flash_attn=self.flash_attn
        )

        print(f'bsize {self.batch_size}')
        print(self.model)
        print(self.model.config.max_position_embeddings)
        print('BWAAA')

        # load and process archive puzzles into strings
        self.archive_name = self.archive_dataset_name
        # with open(self.archive_dataset_name, 'r') as f:
        #     puzzle_archive = json.load(f)
        puzzle_archive = utils.load_dataset_progress(self.archive_name)
        self.archive_puzzle_strs = [utils.make_puzzle(p, self.use_docstring)
                                    for p in puzzle_archive if p['sol_bodies']]
        self.archive_sol_strs = [utils.make_solution(p) for p in puzzle_archive if p['sol_bodies']]
        self.solutions_tokenized = None  # will be populated after filtering

        # load reference probsol
        if self.reference_probsol is None:
            if self.use_docstring:
                self.ref_puzzle = utils.REF_PUZZLE.replace('def sat(', 'def f(')
            else:
                self.ref_puzzle = utils.REF_PUZZLE_NODOC.replace('def sat', 'def f(')
        self.ref_solution = utils.REF_SOL.replace('def sol', 'def g(')

        self.prompt_text = load_prompt_PP(one_shot_prompt_id)
        # with open(os.path.join(os.getcwd(), 'quality_metrics', 'dataset_progress', one_shot_prompt_id), 'r') as f:
        # with open(os.path.join(os.path.dirname(__file__),'quality_metrics', 'dataset_progress', one_shot_prompt_id), 'r') as f:
        #     self.prompt_text = f.read()
            

        self._filter_puzzles()
        self.original_losses = self._get_original_losses()
        print(f'original losses {self.original_losses}')
        pass

    def _filter_puzzles(self, tolerance=800, num_max_tokens=2048):
        print('Filtering long puzzles in the archive')
        if num_max_tokens is None:
            num_max_tokens = self.model.config.max_position_embeddings
        archive_puzzle_sols = [
            self.prompt_text.format(
                puzzle=self.ref_puzzle,
                solution=self.ref_solution,
                archive_puzzle=apuz,
                archive_solution=asol)
            for apuz, asol in zip(self.archive_puzzle_strs, self.archive_sol_strs)]

        archive_tokenized_puzzles = self.tokenizer(archive_puzzle_sols)
        indices_to_keep = []

        for i, ts in enumerate(archive_tokenized_puzzles.input_ids):
            if len(ts) + tolerance < num_max_tokens:
                indices_to_keep.append(i)

        self.archive_puzzle_strs = [self.archive_puzzle_strs[i] for i in indices_to_keep]
        self.archive_sol_strs = [self.archive_sol_strs[i] for i in indices_to_keep]
        self.solutions_tokenized = self.tokenizer(self.archive_sol_strs)
        print("end filtering")

    def _get_original_losses(self):
        # try to load values based on the archive dataset
        path = os.path.join('quality_metrics', 'dataset_progress', 'loss_cache', self.archive_name + '.pt')
        if os.path.exists(path):
            return torch.load(path)
        else:
            # compute the values and cache them (for future runs)
            return self._get_losses(self.ref_puzzle, self.ref_solution)

    def _get_losses(self, puzzle: str, solution: str):
        # format prompts with archive and ref puzzles
        archive_puzzle_sols = [
            self.prompt_text.format(
                puzzle=puzzle,
                solution=solution,
                archive_puzzle=apuz,
                archive_solution=asol)
            for apuz, asol in zip(self.archive_puzzle_strs, self.archive_sol_strs)]

        archive_tokenized_puzzles = self.tokenizer(archive_puzzle_sols, return_tensors='pt', padding=True)

        # print(archive_puzzle_sols[0])

        # get solution mask
        solution_attention_mask = utils.get_solution_mask_from_str_loop(
            full_prompts=archive_puzzle_sols,
            solutions=self.archive_sol_strs,
            tokenizer=self.tokenizer,
            num_solution_tokenss=[len(t) - 1 for t in self.solutions_tokenized.input_ids],
            archive_attention_mask=archive_tokenized_puzzles.attention_mask,
            offsets=[l.tolist().index(1) for l in archive_tokenized_puzzles.attention_mask],
        )
        archive_tokenized_puzzles.loss_attention_mask = solution_attention_mask

        return get_solution_logprobs(archive_tokenized_puzzles, self.model, batch_size=self.batch_size)

    def fitness(self, probsol: P3ProbSolResult, use_pass_k=False) -> float:
        solving_fitness = super().fitness(probsol, use_pass_k)
        if solving_fitness <= 0:
            return solving_fitness  # we require that the problem be solvable by chatgpt

        # check the docstring works fine
        puzzle, solution = utils.parse_puzzle_from_str(probsol.program_str)

        final_losses = self._get_losses(puzzle, solution)

        differences = final_losses - self.original_losses
        fitness = differences.mean().item()
        return - fitness

    def multiple_fitness(self,list_probsol: list[P3ProbSolResult], use_pass_k = False, parrallel_fitness=True, disable_tqdm=True):
        
        list_solving_fitness = super().multiple_fitness(list_probsol, use_pass_k)
        assert len(list_solving_fitness) == len(list_probsol)

        for idx,solving_fitness in enumerate(tqdm(list_solving_fitness,disable=disable_tqdm)):
            if solving_fitness <= 0:
                continue
            else:
                # check the docstring works fine
                puzzle, solution = utils.parse_puzzle_from_str(list_probsol[idx].program_str)

                final_losses = self._get_losses(puzzle, solution)

                differences = final_losses - self.original_losses
                fitness = differences.mean().item()
                list_solving_fitness[idx] = - fitness
                list_probsol[idx].fitness = - fitness
        return list_solving_fitness
