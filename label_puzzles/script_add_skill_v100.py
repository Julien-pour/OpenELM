script='''#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --job-name=v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/out{out1}-%a.out
#SBATCH --error=./out/out{out2}-%a.out

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
index=$SLURM_ARRAY_TASK_ID

conda deactivate
module purge
module load cpuarch/amd
module load python/3.11.5
conda activate codegpt


cd /gpfswork/rech/imi/uqv82bm/OpenELM/label_puzzles/
seed=$SLURM_ARRAY_TASK_ID

python label_formated_data.py --path={path}
kill $SERVER_PID
'''
list_path=["/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-10.json",
 "/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-11.json", "/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-12.json","/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-13.json"]
import os
import argparse
import subprocess
from datetime import datetime

for id,path in enumerate(list_path):
    name=f'vllm41_v100_{path}'
        
    script_formated = script.format(out1=str(id),out2=str(id),path=path)
    extra_path=name
    slurmfile_path = f'slurm/run_vvv100inf'+str(id)+'.slurm'
    with open(slurmfile_path, 'w') as f:
        f.write(script_formated)
    subprocess.call(f'sbatch {slurmfile_path}', shell=True)
