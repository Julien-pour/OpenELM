import os
import argparse
import subprocess
from datetime import datetime

script_1="""#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00

#SBATCH --output=./out/out-{name}.out
#SBATCH --error=./out/out-{name}.out

export TMPDIR=$JOBSCRATCH
module purge
module load python/3.11.5
full_path=$SCRATCH/hf/Meta-Llama-3-70B-Instruct-GPTQ
conda activate vllm41
MAXWAIT=30
sleep $((RANDOM % MAXWAIT))
python -m vllm.entrypoints.openai.api_server --model $full_path --dtype half --api-key token-abc123 --tensor-parallel-size 4 --max-model-len 8000 --gpu-memory-utilization 0.6 &
SERVER_PID=$!

sleep 100

conda deactivate
module purge
module load python/3.11.5

conda activate codegpt



cd /gpfswork/rech/imi/uqv82bm/OpenELM/
python heritability.py --config_name {config_name} --num_puz {num_puz}
"""
# rd_gen
# elm
# elm_nlp
# aces 
# aces_smart
# aces_smart_diversity
# aces_smart_elm
# aces_smart_elm_diversity
list_config=["rd_gen","elm","elm_nlp","aces","aces_smart","aces_smart_diversity","aces_smart_elm","aces_smart_elm_diversity"]
num_puz=100
for config_name in list_config:
    name=f'vllm41_{config_name}'
        
    script_formated = script_1.format(name=name,config_name=config_name,num_puz=num_puz)
    extra_path=name
    slurmfile_path = f'slurm/run_v100inf'+extra_path+'.slurm'
    with open(slurmfile_path, 'w') as f:
        f.write(script_formated)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)
