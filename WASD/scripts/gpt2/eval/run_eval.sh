#!/bin/bash
# Usage (from the repository root):
#   bash ./scripts/gpt2/eval/run_eval.sh <checkpoint> [MASTER_PORT]
#
# <checkpoint> is either a checkpoint directory  results/gpt2/train/<run>/<step>
# or a training run directory                     results/gpt2/train/<run>
# (the step with the best validation ROUGE-L in <run>/log.txt is selected).
# Absolute paths and paths with or without the "results/gpt2/train/" prefix are accepted.

ckpt=${1:?"usage: bash ./scripts/gpt2/eval/run_eval.sh <checkpoint> [MASTER_PORT]"}
MASTER_PORT=${2-29500}
TRAIN_ROOT="results/gpt2/train"

# normalize to a path relative to results/gpt2/train/
ckpt=${ckpt%/}
case "${ckpt}" in
    *"${TRAIN_ROOT}/"*) ckpt=${ckpt#*"${TRAIN_ROOT}/"} ;;
esac

# a run directory: pick the checkpoint with the best validation ROUGE-L
if [[ ! "$(basename "${ckpt}")" =~ ^[0-9]+$ ]]; then
    step=$(python ./tools/find_best_ckpt.py "${TRAIN_ROOT}/${ckpt}") || exit 1
    ckpt="${ckpt}/${step}"
fi

if [ ! -d "${TRAIN_ROOT}/${ckpt}" ]; then
    echo "checkpoint not found: ${TRAIN_ROOT}/${ckpt}"
    exit 1
fi
echo "Evaluating ${TRAIN_ROOT}/${ckpt}"

for seed in 10 20 30 40 50
do
    bash ./scripts/gpt2/eval/eval_main_dolly.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    bash ./scripts/gpt2/eval/eval_main_self_inst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    bash ./scripts/gpt2/eval/eval_main_vicuna.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    bash ./scripts/gpt2/eval/eval_main_sinst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
    bash ./scripts/gpt2/eval/eval_main_uinst.sh ./ ${MASTER_PORT} 1 ${ckpt} --seed $seed  --eval-batch-size 16
done

# mean / std of ROUGE-L over the seeds -> results/gpt2/eval_main/_summary/
python parse_result.py --model gpt2 --method "${ckpt}"
