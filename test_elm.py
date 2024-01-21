import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from openelm import ELM
import os
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(
    # config_path='conf',
    config_name="elmconfig",
    version_base="1.2",
)
def main(config):
    config.output_dir = HydraConfig.get().runtime.output_dir
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    elm = ELM(config)
    # print(
    #     "Best Individual: ",
    #     elm.run(init_steps=config.qd.init_steps, total_steps=config.qd.total_steps),
    # )


if __name__ == "__main__":
    main()
