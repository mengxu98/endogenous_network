#!/bin/bash

# Hill coefficient optimization
python src/hill_coefficient_optimizer.py \
    results/0/network_without_0.txt \
    -o results/hill_coefficient_analysis

# Stability analysis
python src/hill_stability_analysis.py \
    results/0/network_without_0.txt \
    --runs 50 \
    --output results/hill_stability_analysis
