#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=aces_elm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --array=1

#SBATCH --hint=nomultithread
#SBATCH --time=15:00:00

#SBATCH --output=./out/aces_elm_big-%A_%a.out
#SBATCH --error=./out/out_finetune_deep-%A_%a.out
export TMPDIR=$JOBSCRATCH
module purge
module load python/3.11.5
list_model_names=("Mistral-Large-Instruct-2407-AWQ" "Meta-Llama-3.1-405B-Instruct-AWQ-INT4")
index=$SLURM_ARRAY_TASK_ID

model_names_id=${list_model_names[$index]}

full_path=$SCRATCH/hf/$model_names_id
conda activate vllm532  # dont forget to
MAXWAIT=20
sleep $((RANDOM % MAXWAIT))
python -m vllm.entrypoints.openai.api_server --model $full_path --api-key token-abc123 --tensor-parallel-size 8 --max-model-len 6000 &
SERVER_PID=$!

# Wait for the server to be ready
list_sleep=(1000 1800)
sleep ${list_sleep[$index]}

conda deactivate
module purge
module load python/3.11.5

conda activate codegpt2

cd /gpfswork/rech/imi/uqv82bm/OpenELM/
seed=1
path="/gpfswork/rech/imi/uqv82bm/OpenELM/logs/elm/Meta-Llama-3.1-405B-Instruct-AWQ-INT4/aces_elm_seed-1/24-07-29_00:36/step_15" 
python run_elm.py --config-name=aces_elm seed=$seed env.seed=$seed qd.seed=$seed model.model_path=$full_path model_name=$model_names_id
kill $SERVER_PID

# start from save: 'qd.loading_snapshot_map=True' 'qd.log_snapshot_dir=$path'