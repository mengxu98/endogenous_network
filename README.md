# Network Analysis Pipeline

A comprehensive pipeline for network analysis, attractor detection, and stability assessment.

## Prerequisites

### Required Files
Place these files in the `src` directory:
- `network_rewiring.py`
- `network_frame.py`
- `attractor_analysis.py`
- `hill_coefficient_optimizer.py`
- `stability_analysis.py`

### Required Python Packages
- scipy
- numpy
- pandas
- itertools

## Directory Structure

```
network_analysis/
├── src/
│   ├── network_rewiring.py
│   ├── network_frame.py
│   ├── attractor_analysis.py
│   ├── hill_coefficient_optimizer.py
│   └── stability_analysis.py
```

## Usage

### 1. Basic Network Analysis
Run the main analysis pipeline:

```bash
./network_analysis.sh
```

### 2. Advanced Analysis

#### Hill Coefficient Optimization
Run the hill coefficient analysis:

```bash
./hill_optimizer.sh
```

This will:
- Optimize hill coefficients for the network
- Perform stability analysis with 50 simulation runs

Results will be saved in:
- `results/hill_coefficient_analysis/`
- `results/hill_stability_analysis/`

## Input File Formats

### Network File Format
The network file (`network_without_0.txt`) should be formatted as follows:
```
NodeID  Value
1       0.0
2       0.0
...     ...
N       0.0
```
- NodeID: Integer starting from 1
- Value: Float number representing node state

### Gene Mapping File Format
The gene mapping file (`gene_mapping.txt`) should contain:
```
GeneID  GeneName
1       GENE1
2       GENE2
...     ...
N       GENEN
```

## Output Description

### Attractor Analysis Results
The attractor analysis (`results/attractors.txt`) contains:
```
Size    AttractorState
691     ['0.0000', '1.0000', ..., '0.0000']
128     ['0.0000', '1.0000', ..., '0.0000']
```
- Size: Integer representing attractor size
- AttractorState: Array of node states

### Hill Coefficient Analysis
Results include:
- Optimal hill coefficients for each node
- Stability metrics across multiple runs
- Visualization plots of network dynamics

## Advanced Configuration

### Modifying Simulation Parameters
Edit the shell scripts to customize:
- Number of simulation runs
- Output directories
- Analysis parameters

Example:
```bash
# Modify stability analysis parameters
python src/stability_analysis.py \
    results/0/network_without_0.txt \
    --runs 50 \            # Increase number of runs
    --output custom_output  # Change output directory
```
