#!/bin/bash
#SBATCH -C sirocco 
#SBATCH --nodelist=sirocco[14-24]
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=aces_quality
#SBATCH --array=0-4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./out/aces_quality-%j.out
#SBATCH --error=./out/aces_quality-%j.out


cd /projets/flowers/julien/OpenELM/label_puzzles
conda deactivate
conda activate codegpt

python add_quality.py -id $SLURM_ARRAY_TASK_ID