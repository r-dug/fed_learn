#!/bin/bash
#SBATCH --job-name="fed_learn_run"
#SBATCH --partition="gpu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --output=/home/rcdoug03/fed_learn/logs/run.out
#SBATCH --error=/home/rcdoug03/fed_learn/logs/run.err
cd /home/rcdoug03/fed_learn
source /home/rcdoug03/miniconda3/etc/profile.d/conda.sh
wait
conda activate federated-learning
wait
python run.py --overwrite