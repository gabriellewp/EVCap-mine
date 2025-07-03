#!/bin/sh

#SBATCH --job-name=tr_evflm_dist    # The job name.
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=5-00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/gpoerwa/EVCap/output/%x-%J.log
#SBATCH -p performance
#SBATCH --ntasks-per-node=2      # Run 2 tasks on this node (one per GPU)
#SBATCH --gres=gpu:2             # Request 2 GPUs

source ~/.bashrc
micromamba activate base
echo "activate the micromamba env"

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345

# Set proper environment variables for PyTorch distributed
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Print distributed training info
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"

# Launch your script with proper distributed training parameters
# Pass the world_size explicitly to ensure it matches the number of GPUs


echo "This job has $SLURM_NTASKS total tasks"
echo "This job has $SLURM_NTASKS_PER_NODE tasks per node"
echo "This job is running on $SLURM_JOB_NUM_NODES nodes"

# export HF_HOME="/ivi/ilps/personal/gpoerwa/.cache/huggingface" \
# export HF_DATASETS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/datasets" \
# export TRANSFORMERS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/models" \
# export REPLICATE_API_TOKEN="r8_OVbiMFTFdgL07Vzsx876nWU5ObiyHs11iLUZc" \
export HF_TOKEN="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" \
# export PYTHONPATH="/ivi/ilps/personal/gpoerwa/MiniGPT-4:$PYTHONPATH" \
# export TORCH_HOME="/ivi/ilps/personal/gpoerwa/.cache/torch" \
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# Now run your training command

echo "Current PYTHONPATH: $PYTHONPATH"
echo "done setting up environments"

# Record start time before running the training
start_time=$(date +%s)

srun python /home/gpoerwa/EVCap/train_evcap_flowmatching.py \
    --world_size $WORLD_SIZE \
    --use_attention_mask \
    --lambda_flow 0.5 \
    --epochs 5 \
    --bs 4
# python3 searchid.py $START $END
# python3 main.py 
# python3 testing_time.py
# python3 models_inferencing.py
# NUM_GPUS=$SLURM_GPUS_ON_NODE

# # Check if NUM_GPUS was successfully extracted and is a number
# if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
#     echo "Error: Could not determine number of GPUs from SBATCH directive in $0."
#     echo "Defaulting NUM_GPUS to 1. Check the #SBATCH --gres=gpu:N line."
#     NUM_GPUS=1 # Fallback to 1, or use 'exit 1' to stop the script
# fi

# torchrun --nproc-per-node $NUM_GPUS train.py --cfg-path train_configs/minigpt4_llama2_stage2_finetune.yaml

#monitoring with nvidia-smi

# GPU_LOG="gpu_metrics.csv"
# echo "timestamp, gpu_utilization (%), gpu_memory_usage (MiB)" > $GPU_LOG
# (
#     while true; do
#         TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
#         nvidia-smi --query-gpu=utilization.gpu,memory.used \
#                    --format=csv,noheader,nounits \
#                    | awk -v ts="$TIMESTAMP" '{print ts ", " $0}' >> $GPU_LOG
#         sleep 1
#     done
# ) &

# Training is already launched above with srun
# Commenting out this line as it would cause the script to run twice
# python3 train_evcap_flowmatching.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"