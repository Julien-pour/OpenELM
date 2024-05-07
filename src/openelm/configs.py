from dataclasses import dataclass, field
from typing import Any, Optional, Union

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

import transformers


@dataclass
class BaseConfig:
    output_dir: str = "logs/"
    seed: Optional[int] = 0
    model_name: Optional[str] = "default"
@dataclass
class ModelConfig(BaseConfig):
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    deterministic: bool = False
    top_p: float = 1.
    temp: float = 1.0
    gen_max_len: int = -1 # -1 for no limit
    batch_size: int = 64
    model_type: str = "openai"  # Can be "hf", "openai", etc
    model_path: str = "/gpfsscratch/rech/imi/uqv82bm/hf/Meta-Llama-3-70B-Instruct-GPTQ"#"gpt-35-0125"#"gpt-3.5-turbo"  # Can be HF model name or path to local model
    vllm: str = True
    azure: bool = True
    azure_endpoint: str = "https://petunia-3.openai.azure.com/"
    api_version = "2024-02-15-preview", 
    parrallel_call: bool = True # if True, use parallel call to API
    processes: int = 80
    logits_only: bool = False
    do_sample: bool = True
    num_return_sequences: int = 1
    trust_remote_code: bool = True  # needed for mosaicml/mpt-7b-instruct
    request_timeout: int = 120 # timeout for API call
    max_retries: int = 100 # number of retries for API call
@dataclass
class PromptModelConfig(ModelConfig):
    request_timeout: int = 30 # timeout for API call
    model_name: str = "prompt"
    # model_path: str = "gpt-35-0125"#"	"gpt-3.5-turbo-0301"  "gpt-3.5-turbo" #"Salesforce/codegen-350M-mono"


@dataclass
class DiffModelConfig(ModelConfig):
    model_name: str = "diff"
    model_path: str = "CarperAI/diff-codegen-350m-v2"


@dataclass
class QDConfig(BaseConfig):
    """

    """
    model_path: str = "gpt-35-0125" # just for register the model
    init_steps: int = 0  #250 # only mutation with base prompt, then sample from map and mutation after init_steps
    total_steps: int = 1000  #256 #2500
    history_length: int = 4096  #128 #2048
    save_history: bool = False
    save_snapshot_interval: int = 20 #5
    loading_snapshot_map: bool = False  # load map located at log_snapshot_dir
    log_snapshot_dir: str ="" #"/home/flowers/work/OpenELM/logs/elm/env=p3_probsol_Chat_IMGEP_smart/24-02-16_16:11/step_80"#"/home/flowers/work/OpenELM/logs/elm/env=P3ProbSolChatEnv_ELM_NLP/24-02-15_22:14/step_15"# imgep smart "/media/data/flowers/OpenELM/logs/elm/env=p3_probsol_Chat_IMGEP_smart/23-09-14_15:26/step_260" imgep rd: "/media/data/flowers/OpenELM/logs/elm/env=p3_probsol_Chat_IMGEP_random/23-09-14_15:55/step_200"
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = False
    crossover_parents: int = 2
    save_bad_individual: bool = True
    sampling_strategy: str = 'soft_normalised'  # one of {'prob_best_5', 'uniform','soft_normalised'} 
    temperature_soft_sampling: float = 0.2 # temperature for soft_normalised
    n_fewshot_examples: int = 3 # number of example to give to the model before generation
    unique_id: str = "default"

@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (2,)) # 2 for P3 NLP space

@dataclass
class MAPElitesConfig_random(QDConfig): # without mutation just random gen
    qd_name: str = "mapelites"
    init_steps: int = 500
    total_steps: int = 500
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (2,)) # 2 for P3 NLP space
    

@dataclass
class MAPElitesQualityConfig(MAPElitesConfig):
    sampling_strategy: str = 'soft_normalised'  # one of {'prob_best_5', 'uniform','soft_normalised'} 


@dataclass
class CVTMAPElitesConfig(QDConfig):
    qd_name: str = "cvtmapelites"
    n_niches: int = 1024
    cvt_samples: int = 40000
    load_centroids: bool = False # load centroids from log_snapshot_dir

@dataclass
class CVTMAPElitesQualityConfig(CVTMAPElitesConfig):
    sampling_strategy: str = 'soft_normalised'  # one of {'prob_best_5', 'uniform','soft_normalised'} 


@dataclass
class EnvConfig(BaseConfig):
    timeout: float = 10.0  # Seconds
    sandbox: bool = False
    sandbox_server: str = "http://localhost:5000"
    processes: int = 80
    batch_size: int = 32 #5  # Batch size of MAP-Elites
    env_name: str = MISSING
    debug: bool = False
    n_descriptor: int = 20 # number of descriptor for MAP-Elites
    max_descriptor_targeted: int = 5 # max number of target descriptor for MAP-Elites
    IMGEP_mode: str = "none" # guided exploration mode, option: "random" "smart" "none"


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
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    embedding_model_type: str = "hf"  # openai or hf
    embedding_model_path: str = MISSING  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002


@dataclass
class P3ProbSolEnvConfig(EnvConfig):
    env_name: str = "p3_probsol"
    prompt_size: str = "long"  # med or long
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    eval_k: int = 1 #100  # k for pass@k for fitness
    embedding_model_type: str = "openai"  # openai or hf
    embedding_model_path: str = "text-embedding-ada-002"  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002
    model_name: str = "openai" # model used for mutation
    


@dataclass
class P3ProbSolChatEnvConfig_Base(EnvConfig):
    """
    IMGEP random:
    embedding_model_type: str = "openai"
    embedding_model_path: str = "ChatGPT"
    IMGEP_mode: str = "random"
    GPT_feedback: bool = True
    
    IMGEP smart: IMGEP random + IMGEP_mode: str = "smart" 
    
    ELM-NLP: IMGEP random + IMGEP_mode: str = "none" 
    
    rd-gen: same as ELM-NLP
    
    ELM:
    embedding_model_type: str = "hf"
    embedding_model_path: str = "Salesforce/codet5p-110m-embedding"
    GPT_feedback: bool = False
    use_preprocessed_trainset_emb = False
    
    """
    env_name: str = "p3_probsol_Chat"
    prompt_size: str = "med"  # med  
    use_preprocessed_trainset: bool = True # use preprocessed trainset for faster loading + add it to the MAP
    use_preprocessed_trainset_emb: bool = True # True if using NLP feedback
    limited_trainset= True # start with few example ~ 130 examples
    timeout: float = 10.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    eval_k: int = 50 #100  # k for pass@k for fitness
    embedding_model_type: str = "openai" #"hf" # "openai" (for NLP "embedding" or just embedding with text-embedding-ada-002) or "hf" 
    embedding_model_path: str = "ChatGPT" # "Salesforce/codet5p-110m-embedding" # remove "embedding" to use chatgpt embedding in NLP space, otherwise standard emb model e.g hf: Salesforce/codet5p-110m-embedding ; openai: text-embedding-ada-002
    model_name: str = "chatgpt" # model used for mutation, not used ? (if not used should be removed from the config) 
    GPT_feedback: bool = True # use GPT for feedback (MapElites)  
    IMGEP_mode: str = "none" # guided exploration mode, option: "random" "smart" "none"
    N_puzzle_to_gen: int = 5 # number of puzzle to generate for one query
    remove_doc = True # can delete that?
    activate_filtering_description = True # use LLM to describe puzzle after generation so it is not bias by skill labeling
    puzzle_filtering = False # filter or not, only work if puzzle activate_filtering_description = True, filtering not very usefull for now


@dataclass
class P3ProbSolChatEnvConfig(P3ProbSolChatEnvConfig_Base):
    """
    IMGEP random:
    embedding_model_type: str = "openai"
    embedding_model_path: str = "ChatGPT"
    IMGEP_mode: str = "random"
    GPT_feedback: bool = True
    
    IMGEP smart: IMGEP random + IMGEP_mode: str = "smart" 
    
    ELM-NLP: IMGEP random + IMGEP_mode: str = "none" 
    
    rd-gen: same as ELM-NLP
    
    ELM:
    embedding_model_type: str = "hf"
    embedding_model_path: str = "Salesforce/codet5p-110m-embedding"
    GPT_feedback: bool = False
    use_preprocessed_trainset_emb = False
    
    """
    env_name: str = "p3_probsol_Chat"
    use_preprocessed_trainset: bool = True # use preprocessed trainset for faster loading + add it to the MAP
    use_preprocessed_trainset_emb: bool = True # True if using NLP feedback
    embedding_model_type: str = "openai" #"hf" # "openai" (for NLP "embedding" or just embedding with text-embedding-ada-002) or "hf" 
    embedding_model_path: str = "ChatGPT" # "Salesforce/codet5p-110m-embedding" # remove "embedding" to use chatgpt embedding in NLP space, otherwise standard emb model e.g hf: Salesforce/codet5p-110m-embedding ; openai: text-embedding-ada-002
    model_name: str = "chatgpt" # model used for mutation, not used ? (if not used should be removed from the config) 
    GPT_feedback: bool = True # use GPT for feedback (MapElites)  
    IMGEP_mode: str = "none" # guided exploration mode, option: "random" "smart" "none"
    
    
@dataclass
class P3ProbSolChatEnv_IMGEP_smart_Config(P3ProbSolChatEnvConfig_Base):

    env_name: str = "p3_probsol_Chat"
    use_preprocessed_trainset: bool = True # use preprocessed trainset for faster loading + add it to the MAP
    use_preprocessed_trainset_emb: bool = True # True if using NLP feedback
    embedding_model_type: str = "openai" #"hf" # "openai" (for NLP "embedding" or just embedding with text-embedding-ada-002) or "hf" 
    embedding_model_path: str = "ChatGPT" # "Salesforce/codet5p-110m-embedding" # remove "embedding" to use chatgpt embedding in NLP space, otherwise standard emb model e.g hf: Salesforce/codet5p-110m-embedding ; openai: text-embedding-ada-002
    model_name: str = "chatgpt" # model used for mutation, not used ? (if not used should be removed from the config) 
    GPT_feedback: bool = True # use GPT for feedback (MapElites)  
    IMGEP_mode: str = "smart" # guided exploration mode, option: "random" "smart" "none"

@dataclass
class P3ProbSolChatEnv_IMGEP_smart_quality_Config(P3ProbSolChatEnv_IMGEP_smart_Config):
    env_name: str = "p3_probsol_Chat_yes"
    batch_size_quality: Optional[int] = 4
    model_or_model_path: str = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    compile: bool = False
    flash_attn: bool = False


@dataclass
class P3ProbSolChatEnv_IMGEP_random_Config(P3ProbSolChatEnvConfig_Base):

    env_name: str = "p3_probsol_Chat"
    use_preprocessed_trainset: bool = True # use preprocessed trainset for faster loading + add it to the MAP
    use_preprocessed_trainset_emb: bool = True # True if using NLP feedback
    embedding_model_type: str = "openai" #"hf" # "openai" (for NLP "embedding" or just embedding with text-embedding-ada-002) or "hf" 
    embedding_model_path: str = "ChatGPT" # "Salesforce/codet5p-110m-embedding" # remove "embedding" to use chatgpt embedding in NLP space, otherwise standard emb model e.g hf: Salesforce/codet5p-110m-embedding ; openai: text-embedding-ada-002
    model_name: str = "chatgpt" # model used for mutation, not used ? (if not used should be removed from the config) 
    GPT_feedback: bool = True # use GPT for feedback (MapElites)  
    IMGEP_mode: str = "uniform" # guided exploration mode, option: "uniform" "smart" "none"


@dataclass
class P3ProbSolChatEnv_IMGEP_random_quality_Config(P3ProbSolChatEnv_IMGEP_random_Config):
    env_name: str = "p3_probsol_Chat_yes"
    batch_size_quality: Optional[int] = 4
    model_or_model_path: str = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    compile: bool = False
    flash_attn: bool = False

@dataclass
class P3ProbSolChatEnv_ELM_Config(P3ProbSolChatEnvConfig_Base):
    """ need to use it with  cvt mapelites """

    env_name: str = "p3_probsol_Chat"
    use_preprocessed_trainset: bool = True # use preprocessed trainset for faster loading + add it to the MAP
    use_preprocessed_trainset_emb: bool = False # True if using NLP feedback
    embedding_model_type: str = "hf" #"hf" # "openai" (for NLP "embedding" or just embedding with text-embedding-ada-002) or "hf" 
    embedding_model_path: str = "Salesforce/codet5p-110m-embedding" # "Salesforce/codet5p-110m-embedding" # remove "embedding" to use chatgpt embedding in NLP space, otherwise standard emb model e.g hf: Salesforce/codet5p-110m-embedding ; openai: text-embedding-ada-002
    model_name: str = "chatgpt" # model used for mutation, not used ? (if not used should be removed from the config) 
    GPT_feedback: bool = False # use GPT for feedback (MapElites)  
    IMGEP_mode: str = "none" # guided exploration mode, option: "random" "smart" "none"

@dataclass
class P3ProbSolChatEnv_ELM_quality_Config(P3ProbSolChatEnv_ELM_Config):
    env_name: str = "p3_probsol_Chat_yes"
    batch_size_quality: Optional[int] = 4
    model_or_model_path: str = "/home/flowers/work/hf/deepseek-coder-1.3b-instruct"#'deepseek-ai/deepseek-coder-1.3b-instruct'
    compile: bool = False
    flash_attn: bool = False


@dataclass
class P3ProbSolChatEnv_ELM_NLP_Config(P3ProbSolChatEnvConfig_Base):
    """ use it with regular mapelites """
    
    env_name: str = "p3_probsol_Chat"
    prompt_size: str = "med"  # med  
    use_preprocessed_trainset: bool = True # use preprocessed trainset for faster loading + add it to the MAP
    use_preprocessed_trainset_emb: bool = True # True if using NLP feedback
    embedding_model_type: str = "openai" #"hf" # "openai" (for NLP "embedding" or just embedding with text-embedding-ada-002) or "hf" 
    embedding_model_path: str = "ChatGPT" # "Salesforce/codet5p-110m-embedding" # remove "embedding" to use chatgpt embedding in NLP space, otherwise standard emb model e.g hf: Salesforce/codet5p-110m-embedding ; openai: text-embedding-ada-002
    model_name: str = "chatgpt" # model used for mutation, not used ? (if not used should be removed from the config) 
    GPT_feedback: bool = True # use GPT for feedback (MapElites)  
    IMGEP_mode: str = "none" # guided exploration mode, option: "random" "smart" "none"


@dataclass
class P3ProbSolChatEnv_ELM_NLP_quality_Config(P3ProbSolChatEnv_ELM_NLP_Config):
    """ use it with regular mapelites """
    env_name: str = "p3_probsol_Chat_yes"
    batch_size_quality: Optional[int] = 4
    model_or_model_path: str = "/home/flowers/work/hf/deepseek-coder-1.3b-instruct"#'deepseek-ai/deepseek-coder-1.3b-instruct'
    compile: bool = False
    flash_attn: bool = False

@dataclass
class P3ProbSolChatEnv_PP_ELM_NLP_Config(P3ProbSolChatEnv_ELM_NLP_Config):
    """
    Prediction Progress version.
    """
    env_name: str = "p3_probsol_Chat_PP"
    batch_size: Optional[int] = 8
    # archive_dataset_name: str = 'puzzles_train_1.json'
    archive_dataset_name: str = 'puzzles_train_1.json' #"/home/flowers/work/OpenELM/puzzles_train_1.json"#
    model_or_model_path: str = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    reference_probsol: Optional[str] = None
    one_shot_prompt_id: str = 'progress_base_example_prompt.md'
    use_docstring: bool = True
    num_workers: int = 12
    compile: bool = False
    flash_attn: bool = False
    num_max_tokens: Optional[int] = 2048



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

"""
IMGEP random: "qd": "mapelites"

IMGEP smart: "qd": "mapelites" 

ELM-NLP: "qd": "mapelites"
rd-gen: "qd": "mapelites"

ELM: "qd": "cvtmapelites" P3ProbSolChatEnv_ELM

"""

defaults_elm = [
    {"model": "prompt"},
    {"qd": "mapelites"}, #mapelites #"cvtmapelites"},
    {"env": "P3ProbSolChatEnv_ELM_NLP"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]

rd_gen = [
    {"model": "prompt"},
    {"qd": "mapelites_rd"}, #mapelites #"cvtmapelites"},
    {"env": "p3_probsol_Chat"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]

elm_base = [
    {"model": "prompt"},
    {"qd": "cvtmapelites"}, #mapelites #"cvtmapelites"},
    {"env": "P3ProbSolChatEnv_ELM"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]
elm_nlp = [
    {"model": "prompt"},
    {"qd": "mapelites"}, #mapelites #"cvtmapelites"},
    {"env": "P3ProbSolChatEnv_ELM_NLP"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]



aces = [
    {"model": "prompt"},
    {"qd": "mapelites"}, #mapelites #"cvtmapelites"},
    {"env": "p3_probsol_Chat_IMGEP_random"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]

aces_smart = [
    {"model": "prompt"},
    {"qd": "mapelites"}, #mapelites #"cvtmapelites"},
    {"env": "p3_probsol_Chat_IMGEP_smart"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]

#quality
elm_quality = [
    {"model": "prompt"},
    {"qd": "cvtmapelites_quality"}, #mapelites #"cvtmapelites"},
    {"env": "P3ProbSolChatEnv_ELM_Yes_quality"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]
elm_nlp_yes_quality = [
    {"model": "prompt"},
    {"qd": "mapelites_quality"}, #mapelites #"cvtmapelites"},
    {"env": "P3ProbSolChatEnv_ELM_NLP_Yes_quality"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]
aces_yes_quality = [
    {"model": "prompt"},
    {"qd": "mapelites_quality"}, #mapelites #"cvtmapelites"},
    {"env": "p3_probsol_Chat_IMGEP_random_Yes_quality"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]

aces_smart_yes_quality = [
    {"model": "prompt"},
    {"qd": "mapelites_quality"}, #mapelites #"cvtmapelites"},
    {"env": "p3_probsol_Chat_IMGEP_smart_Yes_quality"}, # p3_probsol_Chat_IMGEP_smart,p3_probsol_Chat
    "_self_",
]
@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/elm/${model_name}/${hydra.job.config_name}_seed-${seed}/${now:%y-%m-%d_%H:%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_elm)
    model: Any = MISSING
    qd: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None

@dataclass
class Rd_genConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: rd_gen)
    unique_id="rd_gen"
    
@dataclass
class ELM_baseConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: elm_base)
    unique_id="elm"

@dataclass
class ELM_nlpConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: elm_nlp)
    unique_id="elm_nlp"

@dataclass
class ACESConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: aces)
    unique_id="aces"

@dataclass
class ACES_smartConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: aces_smart)
    unique_id="aces_smart"

# quality

@dataclass
class ELM_yesConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: elm_quality)
    unique_id="elm_yes"

@dataclass
class ELM_nlp_yesConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: elm_nlp_yes_quality)
    unique_id="elm_nlp_yes"
@dataclass
class ACES_yesConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: aces_yes_quality)
    unique_id= "aces_yes"
@dataclass
class ACES_smart_yesConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: aces_smart_yes_quality)
    unique_id="aces_smart_yes"

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
    eval_k: int = 1#-1
    eval_timestamp: str = ""  # optionally provide timestamp of run to eval
    # pass@k, otherwise eval with latest run of every problem


def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="sodarace", node=SodaraceEnvConfig)
    cs.store(group="env", name="image_evolution", node=ImageEnvConfig)
    cs.store(group="env", name="string_evolution", node=StringEnvConfig)
    
    cs.store(group="env", name="p3_probsol", node=P3ProbSolEnvConfig)
    cs.store(group="env", name="p3_probsol_Chat_IMGEP_smart", node=P3ProbSolChatEnv_IMGEP_smart_Config)
    cs.store(group="env", name="p3_probsol_Chat_IMGEP_random", node=P3ProbSolChatEnv_IMGEP_random_Config)
    cs.store(group="env", name="P3ProbSolChatEnv_ELM", node=P3ProbSolChatEnv_ELM_Config)
    cs.store(group="env", name="P3ProbSolChatEnv_ELM_NLP", node=P3ProbSolChatEnv_ELM_NLP_Config)
    cs.store(group="env", name="p3_probsol_Chat", node=P3ProbSolChatEnvConfig)
    cs.store(group="env", name="p3_problem", node=P3ProblemEnvConfig)
    cs.store(group="env", name="P3ProbSolChatEnv_PP_ELM_NLP", node=P3ProbSolChatEnv_PP_ELM_NLP_Config)
    # quality
    cs.store(group="env", name="p3_probsol_Chat_IMGEP_smart_Yes_quality", node=P3ProbSolChatEnv_IMGEP_smart_quality_Config)
    cs.store(group="env", name="p3_probsol_Chat_IMGEP_random_Yes_quality", node=P3ProbSolChatEnv_IMGEP_random_quality_Config)
    cs.store(group="env", name="P3ProbSolChatEnv_ELM_NLP_Yes_quality", node=P3ProbSolChatEnv_ELM_NLP_quality_Config)
    cs.store(group="env", name="P3ProbSolChatEnv_ELM_Yes_quality", node=P3ProbSolChatEnv_ELM_quality_Config)



    cs.store(group="env", name="prompt_evolution", node=PromptEnvConfig)
    cs.store(group="env", name="qdaif", node=QDEnvConfig)
    
    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="mapelites_quality", node=MAPElitesQualityConfig)

    cs.store(group="qd", name="mapelites_rd", node=MAPElitesConfig_random)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites_quality", node=CVTMAPElitesQualityConfig)


    cs.store(group="model", name="prompt", node=PromptModelConfig)
    cs.store(group="model", name="diff", node=DiffModelConfig)
    cs.store(name="elmconfig", node=ELMConfig)
    cs.store(name="p3config", node=P3Config)
    cs.store(name="rd_gen", node=Rd_genConfig)
    cs.store(name="elm", node=ELM_baseConfig)
    cs.store(name="elm_nlp", node=ELM_nlpConfig)
    cs.store(name="aces", node=ACESConfig)
    cs.store(name="aces_smart", node=ACES_smartConfig)
    cs.store(name="elm_nlp_quality", node=ELM_nlp_yesConfig)
    cs.store(name="aces_quality", node=ACES_yesConfig)
    cs.store(name="aces_smart_quality", node=ACES_smart_yesConfig)
    cs.store(name="elm_quality", node=ELM_yesConfig)

    return cs


CONFIGSTORE = register_configstore()
