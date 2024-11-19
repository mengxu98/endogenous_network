#!/bin/bash

# Set Python script paths
NETWORK_SCRIPT="src/network_rewiring.py"
ATTRACTOR_SCRIPT="src/network_frame.py"
ANALYSIS_SCRIPT="src/attractor_analysis.py"

# Check if Python scripts exist
if [ ! -f "$NETWORK_SCRIPT" ] || [ ! -f "$ATTRACTOR_SCRIPT" ] || [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Required Python scripts not found"
    exit 1
fi

# Create necessary directories
mkdir -p results
mkdir -p results/0
mkdir -p results/analysis

# Run original network analysis (0)
echo "Running original network analysis..."
if [ ! -f "results/0/network_without_0.txt" ]; then
    echo "Network file not found, running script..."
    python "$NETWORK_SCRIPT" data results 0
fi

# Verify the network file exists
if [ ! -f "results/0/network_without_0.txt" ]; then
    echo "Error: Network file was not created successfully"
    exit 1
else
    echo "Network file created successfully"
fi

# Run attractor analysis
if [ ! -f "results/attractors.txt" ]; then
    echo "Running attractor analysis..."
    python "$ATTRACTOR_SCRIPT" -i results/0/network_without_0.txt -o results/attractors.txt
fi

# Run detailed analysis and visualization
echo "Generating analysis report..."
python "$ANALYSIS_SCRIPT" \
    -i results/attractors.txt \
    -n results/0/network_without_0.txt \
    -g results/gene_mapping.txt \
    -o results/analysis

echo "Analysis complete. Results are available in results/analysis/"
