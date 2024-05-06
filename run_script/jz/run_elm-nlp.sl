#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --job-name=elm_nlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --array=0-1

#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

#SBATCH --output=./out/out_finetune_deep-%A_%a.out
#SBATCH --error=./out/out_finetune_deep-%A_%a.out
module purge
module load python/3.11.5
list_model_names=("CodeQwen1.5-7B-Chat" "Meta-Llama-3-70B-Instruct-GPTQ")
index=$SLURM_ARRAY_TASK_ID

model_names_id=${list_model_names[$index]}

full_path=$SCRATCH/hf/$model_names_id
conda activate vllm41
MAXWAIT=20
sleep $((RANDOM % MAXWAIT))
python -m vllm.entrypoints.openai.api_server --model $full_path --dtype half --api-key token-abc123 --tensor-parallel-size 4 --max-model-len 6000 &
SERVER_PID=$!

# Wait for the server to be ready
list_sleep=(10 120)
sleep ${list_sleep[$index]}
conda deactivate
module purge
module load python/3.11.5

conda activate codegpt

cd /gpfswork/rech/imi/uqv82bm/OpenELM/
seed=1
path="" 
python run_elm.py --config-name=elm_nlp seed=$seed env.seed=$seed qd.seed=$seed model.model_path=$full_path model_name=$model_names_id
# start from save: 'qd.loading_snapshot_map=True' 'qd.log_snapshot_dir=$path'
kill $SERVER_PID
