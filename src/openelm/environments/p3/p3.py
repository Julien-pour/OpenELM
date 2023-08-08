import json
import re
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
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
import torch
from openelm.configs import P3ProblemEnvConfig, P3ProbSolEnvConfig
from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.p3 import (
    P3_IMPORTS,
    P3_PROBLEM_LONG_SEED,
    P3_PROBLEM_MED_SEED,
    P3_PROBSOL_LONG_SEED,
    P3_PROBSOL_MED_SEED,
)
from openelm.environments.p3 import P3_probsol_chat_med_seed,prompt_solve_puzzle_given_f
from openelm.mutation_model import MutationModel
from openelm.sandbox.server.sandbox_codex_execute import ExecResult
from openelm.utils.code_eval import pass_at_k, pool_exec_processes, type_check
from openelm.utils.code_eval import preprocessing_P3,return_f,extract_args_f,return_g,merge_Q_and_A,scrap_f_g

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
    def __init__(self, program_str: str,result_obj: dict, config: P3ProbSolEnvConfig, emb: list= None ):
        """
        Genotype for a programming puzzle problem+solution pair.
        Args:
            program_str: the code for the pair.
            result_obj: the result of the solution.
            config: environment config
        """

        self.program_str = program_str
        self.result_obj = result_obj
        self.config = config
        self.emb = emb
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
            
            elif self.config.env_name == "p3_probsol_Chat" and self.config.embedding_model_type == "hf": 
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
            elif self.config.embedding_model_type == "hf":
                # weird preprocessing 
                pl = pipeline(
                    "feature-extraction", model=self.config.embedding_model_path
                )
                features = np.array(pl(self.program_str))
                return features.max(axis=0).flatten()
            
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

        # Use the first example in the prompt seed as basis for embedding sizes
        i_first = self.prompt_seed.find("assert")
        first_example = self.prompt_seed[:i_first].strip()
        
        first_example ="def f(x): return True\ndef g(x): return True\nassert f(g())==True"
        out = self.to_phenotype(first_example)

        self.genotype_ndim = np.array(out).shape[-1]
        self.genotype_space = np.repeat([[-1, 1]], self.genotype_ndim, axis=0).T
        
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
        if self.config.use_preprocessed_trainset:
            print("loading preprocessed trainset")
            self.preprocess_p3()
                
            


    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def to_phenotype(self,program_str: str):
        """compute embedding of the program"""
        if self.config.embedding_model_type == "openai":
            emb = np.array(
                get_embedding(program_str, engine=self.config.embedding_model_path))
            return emb
        
        elif self.config.env_name == "p3_probsol_Chat" and self.config.embedding_model_type == "hf": 
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
        
    def preprocess_p3(self, split="train",load_embedding=False):
        trainset = preprocessing_P3(split =split, n_token_max=512,load_embedding = load_embedding)
        
        for puz in trainset:
            del puz["f"], puz["g"],puz["attempts"]
            puz["config"] = self.config
            if not load_embedding:
                puz["emb"]=self.to_phenotype(puz["program_str"])
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

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = []
    ) -> dict[str, str]:

        # prompt with prob+sol that is given (one that was the output of a prev mutation)
        if isinstance(code_batch, list):
            # TODO: get nearby genotypes
            code_batch = code_batch
        elif isinstance(code_batch, str):
            code_batch = [code_batch]
        
        list_few_shot_example_phenotypes = list(self.rng.choice(self.all_phenotypes,size=3))
        list_few_shot_example = [pb.program_str for pb in list_few_shot_example_phenotypes]
        prompt_str = self.prompt_seed_function(list_few_shot_example, code_batch)
        # the prev output was f6_2 and g6_2, so now make it f6_1 and g6_1 for the prompt
        # and remove comments (which contain changes from prev f6_1) from new f6_1
        # TODO: pass in the whole object instead of the program_str since it already parsed some of this?
        # i_g = program_str.find("def g(")
        # remove comments with """
        # program_str = program_str[:i_g] + program_str[i_g:]

        # need to change that to sample problem from the selected cell of the archive
        # prompt_str += f"\n\n{program_str}" # f"\n\n{self.new_probsol_preamble}"

        template = f"{P3_IMPORTS}\n"#{self.new_probsol_preamble}"
        return {"prompt": prompt_str, "template": template}

    def generate_programs(self, code_batch: list[str]) -> list[P3ProbSolResult]:
        """Generate new programs with a mutation model and evaluate them."""
        local_scope_exec = False
        _generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec,do_trunc=False
        )
        
        list_pb=[]
        # parse the generated code 
        for gen_prog in _generated_programs:
            split_pb = copy.deepcopy(gen_prog.replace("```python","```").replace("```\n","```").split("```"))
            for idx in range(len(split_pb)):
                if "def f" in split_pb[idx] and "def g" in split_pb[idx]:
                    list_pb.append(split_pb[idx])
        for idx_assert in range(len(list_pb)):
        #     list_pb[idx] = list_pb[idx].split("assert")[0]+"assert f(g()) == True"
            if not "assert f(" in list_pb[idx_assert]:
                list_pb[idx_assert] = list_pb[idx_assert] + "\nassert f(g()) == True"
        generated_programs = list_pb
        
        for idx in range(len(generated_programs)):
            if not P3_IMPORTS in generated_programs[idx]:
                generated_programs[idx] = P3_IMPORTS+ generated_programs[idx]
        
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
                results = pool_exec_processes(
                    generated_programs,
                    func_name="g",
                    timeout=self.config.timeout,
                    processes=self.config.processes,
                    debug=self.config.debug,
                )
            except Exception:
                return self.generate_programs(code_batch)

        results = [
            {"program_str": gen_prog, "result_obj": res_obj, "config": self.config, "emb": self.to_phenotype(gen_prog)}
            for (gen_prog, res_obj) in zip(generated_programs, results)
        ]
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

    def fitness(self, probsol: P3ProbSolResult, use_pass_k = True) -> float:
        """
        Fitness is the inverse of pass@k of the problem func.
        We want a pass@k of >0 so that the problem is reasonably solvable.
        So fitness=0 if unsolved (which is still better than -np.inf).
        Other than that, more difficult (lower pass@k) => higher fitness.
        """
        # if isinstance(probsol.result_obj, ExecResult):
        #     return -np.inf

        # TODO pass@k eval

        if "g(" in probsol.program_str.split("assert f")[1]:
            extract_run_eval_1 = "f"+probsol.program_str.split("assert f")[1]
        else:
            extract_run_eval_1 = "f(*g())"
        extract_run_eval_2 = "f(g())"
        
        eval_code_1 = str(
            f"{probsol.program_str}\n"
            f"def run_eval():\n"
            f"    return {extract_run_eval_1}"
        )
        eval_code_2 = str(
            f"{probsol.program_str}\n"
            f"def run_eval():\n"
            f"    return {extract_run_eval_2}"
        )
        eval_codes =[eval_code_1, eval_code_2]
        # Run code to see if g6_2 solves f6_2
        result = pool_exec_processes(
            eval_codes,
            func_name="run_eval",
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        # if result[0] is True: what  result[0]== True is the problem is solved
            # return -np.inf
        
        # if just one try more like
        if self.config.eval_k<=1 and use_pass_k:
            if result[1] == True:
                # if f(g())== True
                prog = probsol.program_str.split("\nassert f")
                probsol.program_str = prog[0] + "\nassert f(g()) == True\n"
                return 1.0
            elif result[0] == True: 
                return 1.0
            else:
                return -np.inf
            
        # compute pass@k
        else:
            list_new_puzzles = self.try_solving_problem(probsol)
        
            c = 0
            for idx_sol in range(len(list_new_puzzles)):
                p3probsol = list_new_puzzles[idx_sol]
                if self.fitness(p3probsol, use_pass_k = False) == 1.0:
                    
                    probsol.program_str = p3probsol.program_str
                    c+=1
                

            # c = 0
            # for s in solutions:
            #     if p3_problem.evaluate_solution(s) is True:
            #         c += 1

            pak = pass_at_k(len(list_new_puzzles), c, self.config.eval_k)
            return 1 / pak if pak > 0 else 0

    def random(self) -> list[P3ProbSolResult]:
        program_list = [self.construct_prompt() for _ in range(self.config.batch_size)]
        new_probsols = self.generate_programs(program_list)
        return new_probsols

    def mutate(self, probsol_list: list[P3ProbSolResult]) -> list[P3ProbSolResult]:
        probsols = [pb.program_str for pb in probsol_list]
        program_list = list(map(self.construct_prompt, probsols))
        new_probsols = self.generate_programs(program_list)
        return new_probsols
