#!/bin/bash
# Launch 300-epoch ablation study with reduced canonicalizations
# Canons: spielman, spielman_partition, random_fixed, random_augmented, oap, none
# Hidden dims: 16, 128, 512
# k: 3, 6, 12
# Seeds: 0, 1, 2
# Eigval scale: both
# Total: 6 × 3 × 3 × 3 × 2 = 324 runs
# NOTE: This will overwrite old 50-epoch results for matching configs

cd /home/snirhordan/spectral_symmetry

mkdir -p logs

echo "[$(date)] Starting 300-epoch ablation (324 runs)"
echo "[$(date)] GPUs: 2,3,4,5 (4 GPUs)"
echo "[$(date)] Canons: spielman, spielman_partition, random_fixed, random_augmented, oap, none"
echo "[$(date)] Hidden dims: 16, 128, 512"
echo "[$(date)] k values: 3, 6, 12"
echo "[$(date)] Epochs: 300, patience: 25"

python scripts/ablation_nadav_improvements.py \
    --canonicalization spielman spielman_partition random_fixed random_augmented oap none \
    --hidden-dim 16 128 512 \
    --k 3 6 12 \
    --epochs 300 \
    --patience 25 \
    --gpus 2 3 4 5

echo "[$(date)] 300-epoch ablation complete"
echo "[$(date)] Running analysis..."

python scripts/ablation_nadav_improvements.py --analysis-only

echo "[$(date)] All done. Plots in results/nadav_improvements/plots/"
