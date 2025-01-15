#!/bin/bash
#SBATCH --gpus 1          # Request 2 GPU
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlmodels
accelerate launch --config_file "./HF/accelerate/default_config.yaml" ./scOT/train.py --config "./configs/runHp.yaml" --data_path "./data" --checkpoint_path "./ckptsHp" --wandb_run_name "poseidonB-Hp" --finetune_from "camlab-ethz/Poseidon-B" --replace_embedding_recovery