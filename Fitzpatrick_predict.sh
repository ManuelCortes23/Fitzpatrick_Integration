#!/bin/bash
#SBATCH --job-name=Fitzpatrick_predict   # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=manuel.cortes@ufl.edu     # Where to send mail
#SBATCH --account=egn4951
#SBATCH --qos=egn4951
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=32:00:00               # Time limit hrs:min:sec
#SBATCH --output=Fitzpatrick_predict_%j.log # Standard output and error log

pwd; hostname; date

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

#Begin Script
ml singularity

singularity exec --nv pytorch1.5.0.sif python3 predict.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns --batch-size 4 --CUDA_VISIBLE_DEVICES 0

#End Script

date