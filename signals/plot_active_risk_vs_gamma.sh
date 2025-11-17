#!/bin/bash
#SBATCH --job-name=plot_active_risk_vs_gamma
#SBATCH --output=logs/plot_active_risk_vs_gamma_%A_%a.out
#SBATCH --error=logs/plot_active_risk_vs_gamma_%A_%a.err
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --mail-user=kylecm11@byu.edu 
#SBATCH --mail-type=BEGIN,END,FAIL 

signals=("bab" "meanrev" "momentum")

time_starts=(
    "2000-01-01" "2008-09-01" "2013-06-01" "2020-02-15"
)

time_ends=(
    "2000-01-31" "2008-09-30" "2013-06-30" "2020-03-14"
)

num_signals=${#signals[@]}
num_years=${#time_starts[@]}

signal_index=$(( SLURM_ARRAY_TASK_ID / num_years ))
time_index=$(( SLURM_ARRAY_TASK_ID % num_years ))

signal=${signals[$signal_index]}
start=${time_starts[$time_index]}
end=${time_ends[$time_index]}

# Run this in the signals folder

echo "Plotting signal=$signal"
source ../.venv/bin/activate
python3 plot_active_risk_vs_gamma.py "$signal" 50 "$start" "$end"