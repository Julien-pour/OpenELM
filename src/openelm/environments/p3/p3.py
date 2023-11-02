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
from openai.embeddings_utils import cosine_similarity, get_embedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer # maybe this is the pb (bitsandbytes launch message when doing multiprocess)?
import torch
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from openelm.configs import P3ProblemEnvConfig, P3ProbSolEnvConfig
from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.p3 import (
    P3_IMPORTS,
    P3_PROBLEM_LONG_SEED,
    P3_PROBLEM_MED_SEED,
    P3_PROBSOL_LONG_SEED,
    P3_PROBSOL_MED_SEED,
)
from openelm.environments.p3 import P3_probsol_chat_med_seed,prompt_solve_puzzle_given_f,skills_evaluation,label_puzzle_chatgpt,P3_probsol_chat_med_seed_goal_targeted
from openelm.mutation_model import MutationModel
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils.code_eval import pass_at_k, pool_exec_processes, type_check
from openelm.utils.code_eval import preprocessing_P3,just_remove_example_in_docstring,sample_target_skill_smart,sample_fewshot_example
from joblib import Parallel, delayed, parallel_config
import itertools
# from joblib import parallel_config

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
        self.baseline_emb = np.array(
            get_embedding(baseline, engine=self.config.embedding_model_path)
        )

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
            compare_str = self.program_str
            i_assert = compare_str.find("assert")
            if i_assert > -1:
                compare_str = compare_str[:i_assert]
            emb = np.array(
                get_embedding(compare_str, engine=self.config.embedding_model_path)
            )
            return cosine_similarity(emb, self.baseline_emb)
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



class P3ProbSolResult(Genotype):
    def __init__(self, program_str: str,result_obj: dict, config: P3ProbSolEnvConfig, emb: list= None, idx_generation: int=-1,target_skills=None,fitness=None):
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

        else:
            i_f6 = program_str.find("def f6_2(")
            i_g6 = program_str.find("def g6_2(")
            i_assert = program_str.find("assert")
            self.problem_func = self.program_str[i_f6:i_g6].strip()
            self.solution_func = self.program_str[i_g6:i_assert].strip()



    def __str__(self) -> str:
        return self.program_str

    def to_phenotype(self) -> Optional[Phenotype]:
        if not self.emb is None:
            return self.emb
        else: 
            if self.config.embedding_model_type == "openai":
                # Openai backend to get the embedding
                if "embedding" in self.config.embedding_model_type: 
                    # use the embedding model to get the embedding
                    compare_str = (
                        self.program_str
                    )  # TODO: remove comments from f6_2 for diversity measurement
                    i_assert = compare_str.find("assert f")
                    if i_assert > -1:
                        compare_str = compare_str[:i_assert]
                    emb = np.array(
                        get_embedding(compare_str, engine=self.config.embedding_model_path)
                    )
                    return emb
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
            dummy_features = np.array(
            get_embedding(first_example, engine=self.config.embedding_model_path))
            self.genotype_ndim: int = len(dummy_features)
            self.genotype_space = np.repeat([[0, 1]], self.genotype_ndim, axis=0).T
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
        trainset = preprocessing_P3(split ="train", n_token_max=512)
        for puz in trainset:
            del puz["f"], puz["g"],puz["attempts"]
            puz["config"] = self.config
        list_p3 = [P3ProbSolResult(**p) for p in trainset]
        correct_pb=0
        list_incorrect_puzzle = []
        for i,probsol in enumerate(list_p3):
            if isinstance(probsol.result_obj, ExecResult):
                continue
            if isinstance(probsol.result_obj, str):
                eval_code = (
                    f"{probsol.program_str}\n"
                    f"def run_eval():\n"
                    f"    return f('{probsol.result_obj}')"
                )
            else:
                eval_code = (
                    f"{probsol.program_str}\n"
                    f"def run_eval():\n"
                    f"    return f({probsol.result_obj})"
                )
            # Run code to see if g6_2 solves f6_2
            result = pool_exec_processes(
                eval_code,
                func_name="run_eval",
                debug=True
            )
            if result[0] is False:
                
                list_incorrect_puzzle.append(i)
            else: 
                correct_pb+=1
                
        # remove incorrect_puzzle 2 puzzle are not correct need to fix that (534/536)
        for i in list_incorrect_puzzle[::-1]:
            del list_p3[i]
            
        self.archive_P3puzzle = list_p3

            
                
        print("correct pb", correct_pb)
        print("total_pb",len(list_p3))
        
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
        compare to basae prob sol:
        remove the explicit mutation in the prompt (prompt with underscore i_1 i_2) as it guided to much the model
        and it lead to bad diversity of generated problems.
        
        The objective is to generate problem+solution pairs.
        Args:
            config: the config file path or dict.
            mutation_model: the diff model (or alternatives).
            ans_type: answer type
        """
        self.mutation_model = mutation_model
        self.config = config
        print(f" \n\n ======================\n\n ======================\n\n{self.config.IMGEP_mode} \n\n ======================\n\n\n ======================\n\n")
        self.batch_size = self.config.batch_size
        self.seed_index = self.config.starting_seed
        self.rng = np.random.default_rng(self.config.seed)

        if self.config.prompt_size == "long":
            raise ValueError("long prompt no implemented yet ")
        elif self.config.prompt_size == "med":
            self.prompt_seed_function = P3_probsol_chat_med_seed
            self.prompt_seed= self.prompt_seed_function()
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
        if self.config.GPT_feedback: #init openai model with temp = 0 
            cfg: dict = {
                "max_tokens": 1024,
                "temperature": 0.0,
                "top_p": 1.,
                "max_retries":100,
                # TODO: rename config option?
                "model_name": "gpt-3.5-turbo-0613",#"gpt-4-0613",
                "request_timeout": 70
            }
            self.chatGPT = ChatOpenAI(**cfg) 
        # Use the first example in the prompt seed as basis for embedding sizes
        # i_first = self.prompt_seed.find("assert")
        # first_example = self.prompt_seed[:i_first].strip()
        
        first_example ="def f(x,a=1,b=1): return a*x==b \ndef g(x,a=1,b=1): return b/a\nassert f(g())==True"
        _,n_skills = skills_evaluation(first_example)
        self.n_skills = n_skills
        out = self.to_phenotype(first_example)
        if self.config.embedding_model_type == "openai" and not "embedding" in self.config.embedding_model_type: 
            #NLP space
            # prompt,n_skills = skills_evaluation("aaaa")
            # self.n_skills = n_skills
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

    def label_puzzle(self,program_str,n_attempts=0):
        """
        Label a puzzle with the skills it requires
        TODO: add a safeguard if the model hallucinate too much e.g len(category_idx_predicted) > n_skills
        """
        return label_puzzle_chatgpt(self.chatGPT,program_str,n_attempts=0,return_completion=False)
        # prompt,n_skills = skills_evaluation(program_str)
        # if n_attempts > 4: # should not append but just in case
        #     # raise ValueError("too many attempts to label the puzzle")
        #     print("WARNING: too many attempts to label the puzzle")
        #     return [0. for i in range(n_skills)]
        # response = self.mutation_model.model.generate([[HumanMessage(content=prompt)]])
        # response = response.generations[0][0].text    
        # split_completion = response.split("Therefore, the list of indices for the problem is:") # add assert 
        # if len(split_completion) == 2 :#"Skills parsing
        #     if split_completion[1][-1] == ".":
        #         split_completion[1] = split_completion[1][:-1] 
        #     try :
        #         category_idx_predicted = eval(split_completion[1]) 
        #         list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)]
        #         return list_skill
            
        #     except: # if pb when parsing try to fix them
        #         if split_completion[1].count("]")==1:
        #             try:
        #                 category_idx_predicted = eval(split_completion[1].split("]")[0]+"]")
        #                 list_skill = [1. if i in category_idx_predicted else 0. for i in range(n_skills)] 
        #                 return list_skill
        #             except:
        #                 return self.label_puzzle(program_str,n_attempts=n_attempts+1)
        #         else:
        #             return self.label_puzzle(program_str,n_attempts=n_attempts+1)
            
        # else: 
        #     return self.label_puzzle(program_str,n_attempts=n_attempts+1)
    
    def to_phenotype(self,program_str: str):
        """compute embedding of the program"""
        # "regular" embedding
        if self.config.GPT_feedback: 
            #use chatGPT (or GPT model) to get the "embedding" in NLP space
            return self.label_puzzle(program_str)
        
        elif self.config.embedding_model_type == "openai":
            if "embedding" in self.config.embedding_model_type: 
                emb = np.array(
                    get_embedding(program_str, engine=self.config.embedding_model_path))
                return emb
    
        elif self.config.embedding_model_type == "hf": 
            # when the model can't be loaded, with feat-extraction
            if self.config.embedding_model_path =="Salesforce/codet5p-110m-embedding":
                with torch.no_grad():
                    inputs = self.tokenizer.encode(program_str, return_tensors="pt",truncation=True,max_length=512)
                    emb = self.model(inputs)[0]
                return emb.numpy()
            
            elif self.config.embedding_model_type == "hf":
                # weird preprocessing 
                features = np.array(self.pl(program_str))

                return features.mean(axis=0).flatten() # mean pooling
            
        else:
            raise NotImplementedError
        
    def preprocess_p3(self, split="train",load_embedding=True,debug=False):
        """preprocess the trainset of P3 
        load embedding from json files 
        debug give random embedding to the puzzles for debugging purpose
        """
        load_embedding = self.config.use_preprocessed_trainset_emb
        print("start loading p3 trainset into map")
        trainset = preprocessing_P3(split =split, n_token_max=512,load_embedding = load_embedding,debug=debug)
        
        for puz in tqdm(trainset):
            del puz["f"], puz["g"],puz["attempts"]
            puz["config"] = self.config
            if not load_embedding:
                puz["emb"]=self.to_phenotype(puz["program_str"])
                
            
            puz["program_str"] = just_remove_example_in_docstring(puz["program_str"]) # remove ex in docstring 
                
                
        list_p3 = [P3ProbSolResult(**p) for p in trainset]
        
        correct_pb=0
        list_incorrect_puzzle = []
        for i,probsol in enumerate(list_p3):
            if isinstance(probsol.result_obj, ExecResult):
                continue
            if isinstance(probsol.result_obj, str):
                eval_code = (
                    f"{probsol.program_str}\n"
                    f"def run_eval():\n"
                    f"    return f('{probsol.result_obj}')"
                )
            else:
                eval_code = (
                    f"{probsol.program_str}\n"
                    f"def run_eval():\n"
                    f"    return f({probsol.result_obj})"
                )
            # Run code to see if g6_2 solves f6_2
            result = pool_exec_processes(
                eval_code,
                func_name="run_eval",
                debug=True
            )
            if result[0] is False:
                
                list_incorrect_puzzle.append(i)
            else: 
                correct_pb+=1
                
        # remove incorrect_puzzle 2 puzzle are not correct need to fix that (534/536)
        for i in list_incorrect_puzzle[::-1]:
            del list_p3[i]
            
        self.archive_P3puzzle = list_p3

        print("correct pb", correct_pb)
        print("total_pb",len(list_p3))


    def mutate_vec(self, vec2mutate,k=1):
        """
        take an vector and mutate k values randomly (from 0. to 1. or 1. to 0.)
        """
        vec_mutate=copy.deepcopy(vec2mutate)
        idx = self.rng.choice(vec_mutate.shape[0], k, replace=False)
        vec_mutate[idx] = 1.-vec_mutate[idx]
        return vec_mutate
    
    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = [],random: bool =False
    ) -> dict[str, str]:
        """
        code batch only used for non guided search, it is used  
        """
        n_few_shot_example=3 # that are in the prompt to gen puzzle
        # n_few_shot_example_from_trainset =  1
        # n_few_shot_example_from_archive = n_few_shot_example - n_few_shot_example_from_trainset
        
        skill_targeted=None
        # prompt with prob+sol that is given (one that was the output of a prev mutation)
        if isinstance(code_batch, list):
            # TODO: get nearby genotypes
            code_batch = code_batch
        elif isinstance(code_batch, str):
            code_batch = [code_batch]
        
        if len(code_batch) >=1:
            skill_targeted = code_batch
            
        # choose few shot example
        if random: 
            #random: only use example from trainset
            list_few_shot_example_phenotypes = list(self.rng.choice(self.archive_P3puzzle,size=n_few_shot_example+1))
        else: 
            # use example from archive (and so trainset)
            # list_few_shot_example_phenotypes = list(self.rng.choice(self.archive_P3puzzle,size=n_few_shot_example_from_trainset))
            list_few_shot_example_phenotypes = list(self.rng.choice(self.all_phenotypes,size=n_few_shot_example+1))
            # for puzzz_archive in list_few_shot_example_phenotypes_archive:
            #      list_few_shot_example_phenotypes.append(puzzz_archive)    
            
                 
                 
            
        
        if self.config.IMGEP_mode == "random":
            # target are chosen randomly 
            # choose few shot example that are close in embed space
            
            # target radom skill 
            skill_targeted = np.random.randint(0, 2, self.n_skills,dtype=int).tolist()
            
            
            # choose example that are close in embed space

            
            all_emb=[]
            all_emb_trainset=[]
            
            for puzz in self.all_phenotypes:    
                all_emb.append(puzz.emb)
                
            for puzz in self.archive_P3puzzle:    
                all_emb_trainset.append(puzz.emb)
            all_emb_trainset = np.array(copy.deepcopy(all_emb_trainset))
            
            all_emb = np.array(copy.deepcopy(all_emb))            

            # choose puzzle from closest niches half from trainset
            list_few_shot_example_phenotypes = sample_fewshot_example(skill_targeted, all_emb, self.all_phenotypes, n_few_shot_example=3)
            
            for puzzz in list_few_shot_example_phenotypes: # remove example in doc
                puzzz.program_str=just_remove_example_in_docstring(puzzz.program_str)


            prompt_str = P3_probsol_chat_med_seed_goal_targeted(list_few_shot_example_phenotypes,skill_targeted)
            # skill_targeted.dtype=int
            # check if skill_targeted is a list
            if not isinstance(skill_targeted, list):
                skill_targeted=skill_targeted.tolist()

                        
        elif self.config.IMGEP_mode == "smart":
            # target are chosen smartly + few shot example are chosen smartly
            # target a cell that is close to existing example
            # choose few shot example that are close in embed space
            
            
            
            all_emb=[]
            all_emb_trainset=[]
            
            for puzz in self.all_phenotypes:    
                all_emb.append(puzz.emb)
                
            for puzz in self.archive_P3puzzle:    
                all_emb_trainset.append(puzz.emb)
            all_emb_trainset = np.array(copy.deepcopy(all_emb_trainset))
            all_emb = np.array(copy.deepcopy(all_emb))
            
            # target skill close from explored space
            # Generate all possible binary vectors of dimension 10

            skill_targeted=sample_target_skill_smart(all_emb)

            #old version
            # flag = True
            # while flag: # mutate puzzle until
            #     idx = np.random.choice(all_emb.shape[0], 1, replace=False)[0]
            #     vec = all_emb[idx]
            #     k = self.rng.choice([1,2,3], 1, replace=False,p=[1/3,1/3,1/3]) # change proba distrib? maybe 1/2 1/4 1/4
            #     skill_targeted = self.mutate_vec(vec, k=k)
            #     # check if sampled niched is already filled 
            #     result = np.any(np.all(all_emb == skill_targeted, axis=1))
            #     if not result:
            #         flag=False


            list_few_shot_example_phenotypes= []
            # choose puzzle from closest niches half from trainset
            list_few_shot_example_phenotypes = sample_fewshot_example(skill_targeted, all_emb, self.all_phenotypes, n_few_shot_example=3)

            # remove example given in docstring  
            list_few_shot_example_phenotypes=copy.deepcopy(list_few_shot_example_phenotypes)
            for puzzz in list_few_shot_example_phenotypes:
                puzzz.program_str=just_remove_example_in_docstring(puzzz.program_str)#remove_docstring(puzzz.program_str)
                
            prompt_str = P3_probsol_chat_med_seed_goal_targeted(list_few_shot_example_phenotypes,skill_targeted)
            
        else:
            list_few_shot_example = [pb.program_str for pb in list_few_shot_example_phenotypes]
            skill_targeted=code_batch
            prompt_str = self.prompt_seed_function(list_few_shot_example, code_batch)


        template = f"{P3_IMPORTS}\n"#{self.new_probsol_preamble}"
        return {"prompt": prompt_str, "template": template},skill_targeted


    def generate_programs(self, code_batch: list[dict[str, str]],skill_targeted_list: list[Union[None,list[int]]]) -> list[P3ProbSolResult]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = False
        start_t0 = time.time()
        _generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec,do_trunc=False
        )
        start_t1 = time.time()
        
        list_pb=[]
        # parse the generated code 
        for gen_prog in _generated_programs:
            # should probably use regex (faster)
            split_pb = copy.deepcopy(gen_prog.replace("```python","```").replace("``` python","```").replace("```\n","```").split("```"))
            for idx in range(len(split_pb)):
                if "def f" in split_pb[idx] and "def g" in split_pb[idx]:
                    list_pb.append(split_pb[idx])
                    # if self.config.remove_doc:
                    #     try:
                    #         puzzle_wo_docstring=remove_docstring(split_pb[idx])
                    #         list_pb.append(puzzle_wo_docstring)
                    #     except:
                    #         list_pb.append(split_pb[idx])
                    # else:
                    #     list_pb.append(split_pb[idx])
        for idx_assert in range(len(list_pb)):
        #     list_pb[idx] = list_pb[idx].split("assert")[0]+"assert f(g()) == True"
            if not "assert f(" in list_pb[idx_assert]:
                list_pb[idx_assert] = list_pb[idx_assert] + "\nassert f(g()) == True"
        generated_programs = list_pb
        
        print(f"time to generate {len(generated_programs)} program = {start_t1-start_t0} sec")
        
        list_lib = ["math", "random", "itertools"]
        
        for idx in range(len(generated_programs)):
            if not P3_IMPORTS in generated_programs[idx]:
                generated_programs[idx] = P3_IMPORTS+ generated_programs[idx]
                
            # check if lib are correctly imported (if not import them)
            for lib in list_lib:
                if lib in generated_programs[idx]:
                    if not f"import {lib}" in  generated_programs[idx].split("def f")[0]:
                        generated_programs[idx] = f"import {lib}\n" + generated_programs[idx]
    
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
                # IDK for now if it's usefull to remove assert to have an output even if puzzle is not correct
                # start_t2 = time.time()
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
                # start_t3 = time.time()
                # print(f"time compute return g {len(generated_programs)} program = {start_t3-start_t2} sec")
            except Exception as e:
                results=[]
                for idx_code in range(len(generated_programs)):
                    try:
                        result = pool_exec_processes(
                                generated_programs[idx_code],
                                func_name="g",
                                timeout=self.config.timeout,
                                processes=self.config.processes,
                                debug=self.config.debug,
                            )
                        results.append(result)
                    except Exception as ebis:
                        results.append("Error") #pb when g output a class aaaa:... def g(): return aaaa()
                        
                
        # trick: just label correct problem to save computation time or $$ (chatGPT):
        pre_results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config, "idx_generation": self.idx_generation, "target_skills":target_skills}
            for (gen_prog, res_obj, target_skills) in zip(generated_programs, results, skill_targeted_list)
        ]
        probsol_2_test = [P3ProbSolResult(**p) for p in pre_results]
        start_t4 = time.time()
        list_fitness = [self.fitness(puzz) for puzz in probsol_2_test]
        start_t5 = time.time()
        print( f"time to compute {len(generated_programs)} fitness = {start_t5-start_t4}")
        idx_correct_puzzle = [idx for idx,fit in enumerate(list_fitness) if fit > 0.0]
        print(f"number of correct puzzle {len(idx_correct_puzzle)}")
        list_correct_puzzle = [generated_programs[idx] for idx in idx_correct_puzzle]
        start_t6 = time.time()
        with parallel_config(n_jobs=self.config.processes): #backend='threading', 
            list_phenotype_correct_puzzle = Parallel()(delayed(self.to_phenotype)(puzzl) for puzzl in list_correct_puzzle)
        # list_phenotype_correct_puzzle = Parallel(n_jobs=self.config.processes)(delayed(self.to_phenotype)(puzzl) for puzzl in list_correct_puzzle)
        start_t7 = time.time()
        print( f"time to compute phenotype for {len(list_correct_puzzle)} correct problem  = {start_t7-start_t6}")
        list_phenotype = [[-1] for _ in range(len(generated_programs))] # [-1] when eval is not correct

        # add phenotype of correct puzzle to the list of phenotype
        for idx in range(len(list_phenotype_correct_puzzle)):
            list_phenotype[idx_correct_puzzle[idx]] = list_phenotype_correct_puzzle[idx]
            
        generated_programs = [gen_prog for gen_prog in generated_programs]
        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config, "emb": pheno, "idx_generation": self.idx_generation, "target_skills":target_skills,"fitness":fitness}
            for (gen_prog, res_obj, target_skills,pheno,fitness) in zip(generated_programs, results, skill_targeted_list,list_phenotype,list_fitness)
        ]
        # results = [
        #     {"program_str": gen_prog, "result_obj": res_obj, "config": self.config, "emb": self.to_phenotype(gen_prog), "idx_generation": self.idx_generation, "target_skills":target_skills}
        #     for (gen_prog, res_obj, target_skills) in zip(generated_programs, results, skill_targeted_list,)
        # ]
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
            
            if not P3_IMPORTS in new_pb_str:
                new_pb_str = P3_IMPORTS + new_pb_str
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

        # if "g(" in probsol.program_str.split("assert f")[1]:
        #     extract_run_eval_1 = "f"+probsol.program_str.split("assert f")[1]
        # else: # "error"
        #     extract_run_eval_1 = "f(*g())"
        # extract_run_eval_2 = ""
        
        # eval_code_1 = str(
        #     f"{probsol.program_str}\n"
        #     f"def run_eval():\n"
        #     f"    return {extract_run_eval_1}"
        # )
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

    def random(self) -> list[P3ProbSolResult]:
        # should just take few shot example from trainset not archive
        program_list = []
        skill_targeted_list = []
        for _ in range(self.config.batch_size):
            dic_prompt, skill_targeted = self.construct_prompt(random=True)
            program_list.append(dic_prompt)
            skill_targeted_list.append(skill_targeted)
            
        # program_list = [self.construct_prompt(random=True) for _ in range(self.config.batch_size)]
        new_probsols = self.generate_programs(program_list,skill_targeted_list)
        return new_probsols

    def mutate(self, probsol_list: list[P3ProbSolResult]) -> list[P3ProbSolResult]:
        if self.config.IMGEP_mode == "random" or self.config.IMGEP_mode == "smart":
            program_list = []
            skill_targeted_list = []
            for _ in range(self.config.batch_size):
                dic_prompt, skill_targeted = self.construct_prompt()
                program_list.append(dic_prompt)
                skill_targeted_list.append(skill_targeted)
        else:
            probsols = [pb.program_str for pb in probsol_list]
            # skill_targeted_list = [None for _ in range(len(probsols))]
            program_list_targerted_list = list(map(self.construct_prompt, probsols))
            program_list,skill_targeted_list=[],[]
            for (prgrm_list,skill_targeted) in program_list_targerted_list:
                program_list.append(prgrm_list)
                skill_targeted_list.append(skill_targeted)
            
        new_probsols = self.generate_programs(program_list,skill_targeted_list)

        return new_probsols
