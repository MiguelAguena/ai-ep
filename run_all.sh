#!/usr/bin/env bash
set -euo pipefail

# Default epochs (used if not specified per ratio)
DEFAULT_EPOCHS=1000

# Configure epochs per ratio using associative array
# Format: EPOCHS_PER_RATIO[ratio]=epoch_count
declare -A EPOCHS_PER_RATIO
EPOCHS_PER_RATIO[0.0]=1000
EPOCHS_PER_RATIO[0.1]=900
EPOCHS_PER_RATIO[0.2]=750
EPOCHS_PER_RATIO[0.3]=750
EPOCHS_PER_RATIO[0.4]=500
EPOCHS_PER_RATIO[0.6]=500
EPOCHS_PER_RATIO[0.8]=500

# List of ratios to process
RATIOS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

mkdir -p logs
mkdir -p plots

for r in "${RATIOS[@]}"; do
    # Get epochs for this ratio, or use default
    EPOCHS="${EPOCHS_PER_RATIO[$r]:-$DEFAULT_EPOCHS}"
    
    echo "===================================="
    echo "  DATA_REMOVAL_RATIO = ${r}  (BASE)"
    echo "  EPOCHS = ${EPOCHS}"
    echo "===================================="
    venv/bin/python EP_baseline.py \
        --num_epochs "${EPOCHS}" \
        --data_removal_ratio "${r}" \
        --loss_plot_file "plots/baseline_loss_r${r}.png" \
        > "logs/baseline_r${r}.log"

    echo "===================================="
    echo "  DATA_REMOVAL_RATIO = ${r}  (TIME)"
    echo "  EPOCHS = ${EPOCHS}"
    echo "===================================="
    venv/bin/python EP_time.py \
        --num_epochs "${EPOCHS}" \
        --data_removal_ratio "${r}" \
        --loss_plot_file "plots/time_loss_r${r}.png" \
        > "logs/time_r${r}.log"


    echo "===================================="
    echo "  DATA_REMOVAL_RATIO = ${r}  (AG_AD)"
    echo "  EPOCHS = ${EPOCHS}"
    echo "===================================="
    venv/bin/python EP_aguena_adapted.py \
        --num_epochs "${EPOCHS}" \
        --data_removal_ratio "${r}" \
        --loss_plot_file "plots/aguena_adapted_loss_r${r}.png" \
        > "logs/aguena_adapted_r${r}.log"



    echo "===================================="
    echo "  DATA_REMOVAL_RATIO = ${r}  (TIME Fixed)"
    echo "  EPOCHS = ${EPOCHS}"
    echo "===================================="
    venv/bin/python EP_time_fixed.py \
        --num_epochs "${EPOCHS}" \
        --data_removal_ratio "${r}" \
        --loss_plot_file "plots/aguena_adapted_loss_r${r}.png" \
        > "logs/aguena_adapted_r${r}.log"
done
