#!/bin/bash
#SBATCH --job-name=plot_TE_vs_gamma
#SBATCH --output=logs/plot_TE_vs_gamma_%A_%a.out
#SBATCH --error=logs/plot_TE_vs_gamma_%A_%a.err
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=06:00:00
#SBATCH --mail-user=kylecm11@byu.edu 
#SBATCH --mail-type=BEGIN,END,FAIL 

signals=("bab" "momentum" "meanrev")
signal=${signals[$SLURM_ARRAY_TASK_ID]}
# Run this in the signals folder

echo "Plotting signal=$signal"
source ../.venv/bin/activate
python3 plot_TE_vs_gamma.py "$signal" 50 2020-02-15 2020-03-14