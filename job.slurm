#!/bin/bash
#SBATCH --job-name=oct_run
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --error=oct_run.err
#SBATCH --output=oct_run.out
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=35G
#SBATCH --gres=gpu:4

 
module load anaconda/3
module load cuda/11.4

pip install -r requirements.txt

conda run python3 trainer.py 20 yes yes