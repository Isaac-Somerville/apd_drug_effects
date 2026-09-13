#!/bin/bash
# Simulate APD under drug block.
#
# One array task per unit of work; simulation.simulate_drug_block unpacks
# PBS_ARRAY_INDEX itself. Tasks are independent and safe to rerun: a unit whose
# output already exists is skipped unless --overwrite is passed.
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=48:00:00
#PBS -o logs/ -e logs/
#PBS -N simulate_drug_block
#PBS -J 0-3023

# Repository root and the virtualenv holding requirements.txt.
PUBLIC_REPO="${PUBLIC_REPO:-${HOME}/Public_Data_Simulation}"
VENV="${VENV:-${PUBLIC_REPO}/.venv}"

cd "${PUBLIC_REPO}" || exit 1
mkdir -p logs

source "${VENV}/bin/activate" || exit 1

# Run as a module so that the package root is importable without PYTHONPATH.
python -m simulation.simulate_drug_block "$@"
