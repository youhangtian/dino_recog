export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$1
export TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --standalone --nnodes=1 --nproc_per_node=$2 $3 ${@:4}
