#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=aces_elm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --array=0

#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

#SBATCH --output=./out/wizard-%A_%a.out
#SBATCH --error=./out/wizard-%A_%a.out

# uqv82bm@jean-zay3: jz]$ sbatch run_aces-elm.sl 
# Submitted batch job 986916
# [uqv82bm@jean-zay3: jz]$ sbatch run_aces-elm-
# run_aces-elm-big.sl   run_aces-elm-test.sl  
# [uqv82bm@jean-zay3: jz]$ sbatch run_aces-elm-
# run_aces-elm-big.sl   run_aces-elm-test.sl  
# [uqv82bm@jean-zay3: jz]$ sbatch run_aces-elm-big.sl 
# Submitted batch job 986917
# [uqv82bm@jean-zay3: jz]$ 

export TMPDIR=$JOBSCRATCH
module purge
module load python/3.11.5
list_model_names=("Meta-Llama-3-70B-Instruct-GPTQ" "Meta-Llama-3-70B-Instruct-GPTQ")
index=$SLURM_ARRAY_TASK_ID

model_names_id=${list_model_names[$index]}

full_path=$SCRATCH/hf/$model_names_id
conda activate vllm532  # dont forget to
MAXWAIT=20
sleep $((RANDOM % MAXWAIT))
python -m vllm.entrypoints.openai.api_server --model $full_path --api-key token-abc123 --tensor-parallel-size 4 --max-model-len 6000 &
SERVER_PID=$!

# Wait for the server to be ready
list_sleep=(240 240)
sleep ${list_sleep[$index]}

conda deactivate
module purge
module load python/3.11.5

conda activate codegpt2

cd /gpfswork/rech/imi/uqv82bm/OpenELM/
seed=1
path="" 
python run_elm.py --config-name=wizard_gen seed=$seed env.seed=$seed qd.seed=$seed model.model_path=$full_path model_name=$model_names_id
kill $SERVER_PID

# start from save: 'qd.loading_snapshot_map=True' 'qd.log_snapshot_dir=$path'