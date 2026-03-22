#!/usr/bin/env bash
# Run all BTCV organ classification benchmarks.
#
# Usage:
#   bash scripts/run_benchmarks.sh [GPU_ID]
#
# Default GPU is 0.
#
# Before running:
#   1. Download BTCV data to data/BTCV/  (see data/README.md)
#   2. Download pretrained checkpoints to pretrained/  (see pretrained/README.md)
#   3. Download VoCo weights:  pretrained/VoCo_B_SSL_head.pt
#   4. Download SuPreM weights: pretrained/supervised_suprem_swinunetr_2100.pth

set -euo pipefail

GPU=${1:-0}
SCRIPT="benchmark/scripts/train_classification.py"

echo "===== NEMESIS (frozen) ====="
python ${SCRIPT} --config benchmark/configs/btcv_cls_nemesis.yaml --device_ids ${GPU}

echo "===== NEMESIS (fine-tuned) ====="
python ${SCRIPT} --config benchmark/configs/btcv_cls_nemesis_finetune.yaml --device_ids ${GPU}

echo "===== Random ViT ====="
python ${SCRIPT} --config benchmark/configs/btcv_cls_random_vit.yaml --device_ids ${GPU}

echo "===== ResNet3D-50 ====="
python ${SCRIPT} --config benchmark/configs/btcv_cls_resnet3d.yaml --device_ids ${GPU}

echo "===== VoCo ====="
python ${SCRIPT} --config benchmark/configs/btcv_cls_voco.yaml --device_ids ${GPU}

echo "===== SuPreM ====="
python ${SCRIPT} --config benchmark/configs/btcv_cls_suprem.yaml --device_ids ${GPU}

echo ""
echo "All benchmarks complete. Results saved under results/."
python benchmark/scripts/summarize_results.py --results_dir results/
