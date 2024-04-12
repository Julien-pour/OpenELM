#!/bin/bash
#SBATCH -C sirocco 
#SBATCH --nodelist=sirocco[14-24]
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=elm-nlp
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/elm-nlp-%j.out
#SBATCH --error=./out/elm-nlp-%j.out


cd /projets/flowers/julien/OpenELM/run_script/plafrim/
conda deactivate
conda activate codegpt
cd ..
cd ..
seed=1
path="" 
python run_elm.py --config-name=elm_nlp seed=$seed env.seed=$seed qd.seed=$seed 
# start from save: 'qd.loading_snapshot_map=True' 'qd.log_snapshot_dir=$path'