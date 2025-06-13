#!/bin/bash
### for debugging, you can run the following command to get a shell in the container:
# srun --gpus=1 --nodes=1 --pty /bin/bash

#SBATCH --job-name=pi0-libero-eval
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# export WANDB_API_KEY=981eedf5f279635b72ecbd74304b034fd59288e4
# python3 -m wandb login 

mkdir tmp 2>/dev/null

# Run policy server
uv run scripts/serve_policy.py --env=LIBERO > tmp/slurm-$SLURM_JOB_ID-policy.log 2>&1 &
pid=$!
echo Policy server is running on pid $pid
echo Waiting for 30 sec. to let the policy server start...
sleep 30

# Activate the virtual environment
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero

# Evaluate the policy on the LIBERO environment.
python3 examples/libero/main.py

echo Killing policy server... waiting 10 sec.
sleep 10
kill $pid
