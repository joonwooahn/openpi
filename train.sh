#!/bin/bash
### for debugging, you can run the following command to get a shell in the container:
# srun --gpus=1 --nodes=1 --pty /bin/bash

#SBATCH --job-name=pi0-fast-libero
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# export WANDB_API_KEY=981eedf5f279635b72ecbd74304b034fd59288e4

### Convert your data to a LeRobot dataset (which we use for training)
# uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data

uv run scripts/compute_norm_stats.py --config-name=pi0_fast_libero

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi0_fast_libero --exp-name=jw_pi_0-fast_test --overwrite