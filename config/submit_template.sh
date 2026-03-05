#!/bin/bash

#SBATCH --job-name=lwf_baseline_unet_slurm
#SBATCH --account=pn39sa-c
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=256GB
#SBATCH --output=/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026/di54xen/experiments/LWF-DLR/slurm_logs/%j_log.out
#SBATCH --error=/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026/di54xen/experiments/LWF-DLR/slurm_logs/%j_log.err
# optional:
# #SBATCH --mail-type=END
# #SBATCH --mail-user=your@email.de
# #SBATCH --exclude=i01r04c02s01,i01r04c02s02

# Setup
module load slurm_setup
module load uv
source /dss/dsshome1/02/di54xen/projects/oma24-proj/.venv/bin/activate

# Run training
uv run train_lwf --config /dss/dsshome1/02/di54xen/projects/oma24-proj/config/lwf_template.yaml
