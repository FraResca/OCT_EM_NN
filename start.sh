#!/bin/bash

conda create --name oct python=3.9 && conda activate oct

pip install -r requirements.txt

sbatch job.slurm