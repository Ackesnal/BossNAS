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

PYTHON=${PYTHON:-"python"}

CFG="configs/nats_c100_bs256_accumulate4_gpus4.py"
GPUS=4
PY_ARGS=${@:3}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

srun $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train.py $CFG --work_dir $WORK_DIR --seed 0 --launcher pytorch ${PY_ARGS}
