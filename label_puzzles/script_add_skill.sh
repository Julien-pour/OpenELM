module purge
module load cpuarch/amd
module load python/3.11.5
model_names_id="Meta-Llama-3-70B-Instruct-GPTQ"
full_path=$SCRATCH/hf/Meta-Llama-3-70B-Instruct-GPTQ
conda activate vllm41
MAXWAIT=200
sleep $((RANDOM % MAXWAIT))
python -m vllm.entrypoints.openai.api_server --model $full_path --dtype half --api-key token-abc123 --tensor-parallel-size 8 --max-model-len 8000 --gpu-memory-utilization 0.8 &
SERVER_PID=$!

sleep 100

conda deactivate
module purge
module load cpuarch/amd
module load python/3.11.5

conda activate codegpt


cd /gpfswork/rech/imi/uqv82bm/OpenELM/label_puzzles/
seed=$SLURM_ARRAY_TASK_ID

path="" 
python label_formated_data.py
kill $SERVER_PID
