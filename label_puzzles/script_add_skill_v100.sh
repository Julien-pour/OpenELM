#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --job-name=v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=./out/ou--%a.out
#SBATCH --error=./out/out-%a.out

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
list_path=("/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-10.json" "/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-11.json" "/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-12.json" "/gpfswork/rech/imi/uqv82bm/OpenELM/logs/archives/llama-70/4a100/elm_seed-13.json")
path= ${list_path[$index]}
conda activate codegpt


cd /gpfswork/rech/imi/uqv82bm/OpenELM/label_puzzles/
seed=$SLURM_ARRAY_TASK_ID

path="" 
python label_formated_data.py --path $path
kill $SERVER_PID
