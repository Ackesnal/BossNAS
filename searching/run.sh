#!/usr/bin/bash
#SBATCH -N 1
#SBATCH --job-name=BossNASSearch
#SBATCH -c 4
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:4

module load anaconda/3.7
source activate bossnas

srun bash dist_train.sh configs/nats_c100_bs256_accumulate4_gpus4.py 4