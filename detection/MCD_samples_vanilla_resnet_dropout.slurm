#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --partition=gpu40G,gpuv100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel-alfonso.montoyavasquez@cea.fr

echo "Starting commands!"
pwd;hostname
printf "\n"
printf "load modules: "
module load anaconda
module load cuda/11.6
printf "\n"
printf "Check loaded modules: "
module list


# With Slurm jobs, your ~/.bashrc is not sourced, so it doesn’t initialize the shell for Conda environments
eval "$(conda shell.bash hook)"
conda activate p38_vos_env

echo "list conda-envs explicit paths:"
conda info --envs
printf "\n"
ls /home/users/dmontoya/.conda/envs
printf "\n"

printf "\n"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
printf "\n"

python --version
printf "\n"
# nvidia-smi
# printf "\n"
printf "Start inference:"
srun --kill-on-bad-exit=1 python TDL_get_mcd_and_entropy_samples_bdd_ood.py --dataset-dir ../../datasets/bdd100k --test-dataset bdd_custom_val --config-file BDD-Detection/faster-rcnn/vanilla.yaml --inference-config Inference/TDL_mcd.yaml --random-seed 1 --image-corruption-level 0

# dmesg -T
printf "\n"
echo "Done!"
