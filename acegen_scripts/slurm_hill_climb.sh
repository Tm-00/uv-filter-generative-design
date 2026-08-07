#!/bin/bash

# ==============================================================================
# BEFORE RUNNING: see SETUP.md - replace <YOUR_BASE_PATH> and <YOUR_ACCOUNT>,
# and confirm you have access to University of Birmingham's BlueBEAR HPC
# (this script will not run on a different HPC system without adapting the
# module load lines and venv setup).
# ==============================================================================

#SBATCH --job-name=hill_climb_gpt2
#SBATCH --ntasks=1
#SBATCH --time=3:0:0
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --output=<YOUR_BASE_PATH>/experiments/Slurm_Outputs/s_out/slurm-%j.out
#SBATCH --error=<YOUR_BASE_PATH>/experiments/Slurm_Outputs/s_err/slurm-%j.err

echo "
################################################################################
# Starting batch job ${SLURM_JOB_ID} on $(hostname)
################################################################################
"

export PROJECT_PATH="<YOUR_BASE_PATH>"
${PROJECT_PATH}/templates/create_venv.sh

module purge; module load bluebear
module load bear-apps/2022b
module load Python/3.10.8-GCCcore-12.2.0

export VENV_DIR="<YOUR_BASE_PATH>/venvs"
export VENV_PATH="${VENV_DIR}/venv310-${BB_CPU}"
source ${VENV_PATH}/bin/activate

which python
python --version
python -u <YOUR_BASE_PATH>/code/acegen-chemical/scripts/hill_climb/hill_climb.py --config-path <YOUR_BASE_PATH>/experiments --config-name config_gpt2_uvfilter

echo "finished"
