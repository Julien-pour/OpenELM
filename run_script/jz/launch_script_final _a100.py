import os
import argparse
import subprocess
from datetime import datetime

script_1="""#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=10-13
#SBATCH --output=./out/out-{name}-%a.out
#SBATCH --error=./out/out-{name}-%a.out

export TMPDIR=$JOBSCRATCH
module purge
module load cpuarch/amd
module load python/3.11.5
model_names_id="Meta-Llama-3-70B-Instruct-GPTQ"
full_path=$SCRATCH/hf/Meta-Llama-3-70B-Instruct-GPTQ
conda activate vllm41
MAXWAIT=200
sleep $((RANDOM % MAXWAIT))
python -m vllm.entrypoints.openai.api_server --model $full_path --dtype half --api-key token-abc123 --tensor-parallel-size 4 --max-model-len 8000 --gpu-memory-utilization 0.8 &
SERVER_PID=$!

sleep 100

conda deactivate
module purge
module load cpuarch/amd
module load python/3.11.5

conda activate codegpt


cd /gpfswork/rech/imi/uqv82bm/OpenELM/
seed=$SLURM_ARRAY_TASK_ID

path="" 
python run_elm.py --config-name={config_name} seed=$seed env.seed=$seed qd.seed=$seed model.model_path=$full_path model_name=$model_names_id
kill $SERVER_PID

"""
# rd_gen
# elm
# elm_nlp
# aces 
# aces_smart
# aces_smart_diversity
# aces_smart_elm
# aces_smart_elm_diversity
list_config=["rd_gen","elm","elm_nlp","aces","aces_smart","aces_smart_diversity","aces_smart_elm","aces_smart_elm_diversity","aces_diversity","aces_elm","aces_elm_diversity"]
for config_name in list_config:
    name=f'vllm41_v100_{config_name}'
        
    script_formated = script_1.format(name=config_name,config_name=config_name)
    extra_path=name
    slurmfile_path = f'slurm/run_a100inf'+extra_path+'.slurm'
    with open(slurmfile_path, 'w') as f:
        f.write(script_formated)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)
