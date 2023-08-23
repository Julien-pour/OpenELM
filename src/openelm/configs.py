from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseConfig:
    output_dir: str = "logs/"

@dataclass
class ModelConfig(BaseConfig):
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    deterministic: bool = False
    top_p: float = 1.
    temp: float = 0.7
    gen_max_len: int = -1 # -1 for no limit
    batch_size: int = 3
    model_type: str = "openai"  # Can be "hf", "openai", etc
    model_path: str = "gpt-3.5-turbo-0613"#"gpt-3.5-turbo"  # Can be HF model name or path to local model
    parrallel_call: bool = True # if True, use parallel call to API
    processes: int = 8
    logits_only: bool = False
    do_sample: bool = True
    num_return_sequences: int = 1
    trust_remote_code: bool = True  # needed for mosaicml/mpt-7b-instruct


@dataclass
class PromptModelConfig(ModelConfig):
    model_name: str = "prompt"
    model_path: str = "gpt-3.5-turbo-0613"#"	"gpt-3.5-turbo-0301"  "gpt-3.5-turbo" #"Salesforce/codegen-350M-mono"


@dataclass
class DiffModelConfig(ModelConfig):
    model_name: str = "diff"
    model_path: str = "CarperAI/diff-codegen-350m-v2"


@dataclass
class QDConfig(BaseConfig):
    model_path: str = "gpt-3.5-turbo-0613" # just for register the model
    init_steps: int = 0 #250 # only mutation with base prompt, then sample from map and mutation after init_steps
    total_steps: int = 10 #2500 
    history_length: int = 128
    save_history: bool = False
    save_snapshot_interval: int = 5
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = False
    crossover_parents: int = 2
    save_all_individual: bool = True


@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (2,))


@dataclass
class CVTMAPElitesConfig(QDConfig):
    qd_name: str = "cvtmapelites"
    n_niches: int = 1024
    cvt_samples: int = 42000


@dataclass
class EnvConfig(BaseConfig):
    timeout: float = 5.0  # Seconds
    sandbox: bool = False
    sandbox_server: str = "http://localhost:5000"
    processes: int = 1
    batch_size: int = 5 # 10  # Batch size of MAP-Elites
    env_name: str = MISSING
    debug: bool = True
    seed: Optional[int] = 42


@dataclass
class SodaraceEnvConfig(EnvConfig):
    env_name: str = "sodarace"
    eval_ms: int = 1000  # Milliseconds
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            # Height, Width, Mass dimensions
            [0, 500],
            [0, 500],
            [0, 1000],
        ]
    )
    starting_seeds: list[str] = field(default_factory=lambda: ["square"])
    instruction: int = 2
    crossover: bool = False


@dataclass
class ImageEnvConfig(EnvConfig):
    env_name: str = "image_evolution"
    behavior_mode: str = "3-channel"
    target: str = "circle"


@dataclass
class StringEnvConfig(EnvConfig):
    env_name: str = "string_evolution"
    target: str = "MapElites"


@dataclass
class P3ProblemEnvConfig(EnvConfig):
    env_name: str = "p3_problem"
    prompt_size: str = "long"  # med or long
    timeout: float = 1.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    embedding_model_type: str = "hf"  # openai or hf
    embedding_model_path: str = MISSING  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002


@dataclass
class P3ProbSolEnvConfig(EnvConfig):
    env_name: str = "p3_probsol"
    prompt_size: str = "long"  # med or long
    timeout: float = 1.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    eval_k: int = -1 #100  # k for pass@k for fitness
    embedding_model_type: str = "openai"  # openai or hf
    embedding_model_path: str = "text-embedding-ada-002"  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002
    model_name: str = "openai" # model used for mutation

@dataclass
class P3ProbSolChatEnvConfig(EnvConfig):
    env_name: str = "p3_probsol_Chat"
    prompt_size: str = "med"  # med or long
    use_preprocessed_trainset: bool = True # use preprocessed trainset for faster loading + add it to the MAP
    timeout: float = 1.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    eval_k: int = -1 #100  # k for pass@k for fitness
    embedding_model_type: str = "openai" # "openai" (for NLP "embedding" or just embedding with text-embedding-ada-002) or "hf" 
    embedding_model_path: str = "ChatGPT" # remove "embedding" to use chatgpt embedding in NLP space, otherwise standard emb model e.g hf: Salesforce/codet5p-110m-embedding ; openai: text-embedding-ada-002
    model_name: str = "openai" # model used for mutation, not used ? (if not used should be removed from the config) 
    GPT_feedback: bool = True # use GPT for feedback (MapElites)  
    IMGEP_mode: str = "none" # guided exploration mode, option: "random" "smart" "none"
    
    
@dataclass
class QDEnvConfig(EnvConfig):
    env_name: str = "qdaif"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 5], 
            [0, 5],
        ]
    )


@dataclass
class PromptEnvConfig(EnvConfig):
    env_name: str = "prompt_evolution"
    task_name: str = "antonym"  # toy or antonym or animal or cot
    evals_per_prompt: int = 10
 

# baseline 0 and 1 (give few shot example then gen new pb, and openELM)
# defaults_elm = [
#     {"model": "prompt"},
#     {"qd": "cvtmapelites"}, #"mapelites"},
#     {"env": "p3_probsol_Chat"}, #sodarace"},p3_probsol_Chat
#     "_self_",
# ]

# baseline 2 and 3 (openELM in NLP space, and GPT feedback)
defaults_elm = [
    {"model": "prompt"},
    {"qd": "mapelites"}, #"mapelites"},
    {"env": "p3_probsol_Chat"}, #sodarace"},p3_probsol_Chat
    "_self_",
]


@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/elm/${hydra.job.override_dirname}/${now:%y-%m-%d_%H:%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_elm)
    model: Any = MISSING
    qd: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None


defaults_p3 = [
    {"model": "prompt"},
    {"env": "p3_probsol_Chat"},#p3_probsol_Chat
    "_self_",
]


@dataclass
class P3Config(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/p3/${hydra.job.override_dirname}/${now:%y-%m-%d_%H:%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_p3)
    model: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None
    # --- The below are for run_p3.py
    iterations_per_puzzle: int = 5
    starting_seeds: list[int] = field(
        default_factory=lambda: [1,2,3]
    )  # indices of selection of puzzles to evaluate with
    save_results: bool = True
    save_result_obj: bool = False  # if saving results, include the whole output
    # text from model for each iteration (which can get long)
    probsol: bool = True  # generate new problem+solution pairs from given
    # problems instead of just solutions to given problems
    # set eval_k >0 to evaluate pass@k of previous runs using this k, instead of
    # doing a new run
    eval_k: int = -1#-1
    eval_timestamp: str = ""  # optionally provide timestamp of run to eval
    # pass@k, otherwise eval with latest run of every problem


def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="sodarace", node=SodaraceEnvConfig)
    cs.store(group="env", name="image_evolution", node=ImageEnvConfig)
    cs.store(group="env", name="string_evolution", node=StringEnvConfig)
    cs.store(group="env", name="p3_probsol", node=P3ProbSolEnvConfig)
    cs.store(group="env", name="p3_probsol_Chat", node=P3ProbSolChatEnvConfig)
    cs.store(group="env", name="p3_problem", node=P3ProblemEnvConfig)
    cs.store(group="env", name="prompt_evolution", node=PromptEnvConfig)
    cs.store(group="env", name="qdaif", node=QDEnvConfig)
    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="model", name="prompt", node=PromptModelConfig)
    cs.store(group="model", name="diff", node=DiffModelConfig)
    cs.store(name="elmconfig", node=ELMConfig)
    cs.store(name="p3config", node=P3Config)
    return cs


CONFIGSTORE = register_configstore()
