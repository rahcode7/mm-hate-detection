#!/bin/bash

#SBATCH --job-name=mm-flava
#SBATCH -A research
#SBATCH --output=runs/flava/flava.txt
#SBATCH -n 10
#SBATCH --gres=gpu:3
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='flava'
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='base' 
RUN_TYPE='train' # train,inference
DATE='29Mar'
CHECKPOINT="checkpoints/checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$DATE"
NUM_GPUS=3

if [ "$RUN_TYPE" = "train" ]; then
    mkdir checkpoints
    rm -rf $CHECKPOINT
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=5
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2
    
    
    # Distributed base
    # torchrun --nnodes 1 --nproc_per_node 3 --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900 \    

    # BLIP base + lr
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models/flava/flava-train.py --num_epochs $EPOCHS --train_batch_size 2 --val_batch_size 2 --train_dir datasets/FB-HM/data \
    #     --val_dir datasets/FB-HM/data --checkpoint_dir  $CHECKPOINT  \
    #     --experiment_name ada-$MODEL-$EXP_NAME-$DATE --wandb_status online --accumulation_steps 4 --lr 1

    # torchrun --nnodes 1 --nproc_per_node  $NUM_GPUS --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900 \    
    # python models/flava/flava-train.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/FB-HM/data \
    #     --val_dir datasets/FB-HM/data --checkpoint_dir  $CHECKPOINT  \
    #     --experiment_name ada-$MODEL-$EXP_NAME-$DATE --wandb_status online --accumulation_steps 4 --lr 1


    
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models/flava/flava-train.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/FB-HM/data \
    #      --val_dir datasets/FB-HM/data --checkpoint_dir  $CHECKPOINT  \
    #      --experiment_name ada-$MODEL-$EXP_NAME-$DATE --wandb_status disabled --accumulation_steps 4 --lr 0

    # OCR 
    accelerate launch --multi_gpu --num_processes=$NUM_GPUS models/flava/flava-train-ocr.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/FB-HM/data \
         --val_dir datasets/FB-HM/data --checkpoint_dir  $CHECKPOINT  \
         --experiment_name ada-$MODEL-$EXP_NAME-$DATE --wandb_status disabled --accumulation_steps 4 --lr 1
    

    # MAC single
    #  accelerate launch  --num_processes=$NUM_GPUS models/flava/flava-train.py --num_epochs $EPOCHS --train_batch_size 2 --val_batch_size 2 --train_dir /Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/data \
    #     --val_dir /Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/data --checkpoint_dir  $CHECKPOINT  \
    #     --experiment_name ada-$MODEL-$EXP_NAME-$DATE --wandb_status offline --accumulation_steps 4 --lr 1

elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    #rm -r datasets/results/$MODEL/$EXP_NAME

    # Exp 1 . base 384
    export NUM_NODES=1
    # export EPOCHS=10
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0
    
    python models-hf/flava-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  checkpoints/$CHECKPOINT --question_dir datasets/questions/all/master.json \
        --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME  --machine_type 'dp' --image_size 576 --lr 1

else
    echo "Not valid"
fi 

