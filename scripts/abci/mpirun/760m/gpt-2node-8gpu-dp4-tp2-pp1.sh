#!/bin/bash
#$ -l rt_F=2
#$ -l h_rt=3:00:00
#$ -j y
#$ -o output/
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.6/8.6.0
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

source /home/acf15649kv/work/Megatron-DeepSpeed/.env/bin/activate
cd /home/acf15649kv/work/Megatron-DeepSpeed

# distributed settings
GPUS_PER_NODE=4
NNODES=$NHOSTS
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP_SIZE=2
PP_SIZE=1
DP_SIZE=$(($WORLD_SIZE / ($TP_SIZE * $PP_SIZE)))

echo -e "\nWORLD_SIZE: $WORLD_SIZE, GPUS_PER_NODE: $GPUS_PER_NODE, NNODES: $NNODES"
echo -e "TP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE, DP_SIZE: $DP_SIZE\n"

# dataset, checkpoint path
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=/groups/gcf51174/checkpoints/megatron-deepspeed/mpirun/760m/${NNODES}node-${WORLD_SIZE}gpu-dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}

mkdir -p $CHECKPOINT_PATH

MICRO_BATCHSIZE=8
GLOBAL_BATCHSIZE=$(($MICRO_BATCHSIZE * $DP_SIZE))

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

# hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
cat $SGE_JOB_HOSTLIST > $HOSTFILE_NAME

# model size (760m)
NUM_LAYERS=24
HIDDEN_SIZE=1536
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=2048


mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO  -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python pretrain_gpt.py \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PP_SIZE \
  --num-layers $NUM_LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $NUM_ATTENTION_HEADS \
  --micro-batch-size $MICRO_BATCHSIZE \
  --global-batch-size $GLOBAL_BATCHSIZE \
  --seq-length $SEQ_LENGTH \
  --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
  --train-iters 500000 \
  --lr-decay-iters 320000 \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --checkpoint-activations \
  --log-interval 1 \
  --save-interval 3000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --fp16 \
  --use-mpi \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --wandb-name "gpt2_760m_${NNODES}node_dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun"
