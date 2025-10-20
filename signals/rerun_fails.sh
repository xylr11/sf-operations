#!/bin/bash
#SBATCH --job-name=rerun_failed
#SBATCH --output=logs/low_vol_rerun_failed_%A_%a.out
#SBATCH --error=logs/low_vol_rerun_failed_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --mail-user=kylecm11@byu.edu
#SBATCH --mail-type=FAIL

DATA_PATH="signal_data.parquet"
SCRIPT="get_signal_weights.py"

# Define the failed runs as arrays
SIGNALS=("bab" "bab")
STARTS=("2013-06-27" "2014-06-27")
ENDS=("2014-06-26" "2015-06-26")

# Pick values based on SLURM_ARRAY_TASK_ID
SIGNAL=${SIGNALS[$SLURM_ARRAY_TASK_ID]}
START=${STARTS[$SLURM_ARRAY_TASK_ID]}
END=${ENDS[$SLURM_ARRAY_TASK_ID]}

source ../.venv/bin/activate
echo ">>> Running ${SIGNAL} from ${START} to ${END}"
python "$SCRIPT" "$DATA_PATH" "$SIGNAL" "$START" "$END" --write
