#!/bin/bash
#SBATCH -C sirocco 
#SBATCH --nodelist=sirocco[14-24]
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --job-name=aces_smart_quality
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/aces_smart_quality-%j.out
#SBATCH --error=./out/aces_smart_quality-%j.out


cd /projets/flowers/julien/OpenELM/run_script/plafrim/
conda deactivate
conda activate codegpt
cd ..
cd ..
seed=1
path="" 
python run_elm.py --config-name=aces_smart_quality seed=$seed env.seed=$seed qd.seed=$seed env.model_or_model_path="/projets/flowers/julien/hf/deepseek-coder-1.3b-instruct"
# start from save: 'qd.loading_snapshot_map=True' 'qd.log_snapshot_dir=$path'