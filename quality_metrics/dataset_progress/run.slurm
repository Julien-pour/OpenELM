#!/bin/bash
#SBATCH --account=imi@a100
#SBATCH -C a100
#SBATCH --job-name=progree-metrics
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=outs/progress-metrics-%j.out
#SBATCH --error=outs/progress-metrics-%j.out

cd $WORK/OpenELM

export WANDB_ONLINE_MODE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DATE=$(date +"%Y-%m-%d_%T")

conda deactivate
module purge

conda activate aces

srun python quality_metrics/dataset_progress/progress_metrics.py --batch_size 8