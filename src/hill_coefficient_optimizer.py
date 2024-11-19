# -*- coding: utf-8 -*-
"""
@title Hill Coefficient Optimization
@auther mengxu
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class HillCoefficientOptimizer:
    def __init__(self, network_file, experimental_data=None):
        """
        Initialize Hill coefficient optimizer

        Args:
            network_file: Network structure file path
            experimental_data: Experimental data (if available)
        """
        self.network = pd.read_csv(network_file, sep="\t")
        self.experimental_data = experimental_data
        self.min_hill = 1.0  # Hill coefficient minimum value
        self.max_hill = 20.0  # Hill coefficient maximum value
        self.n_nodes = len(self.network)
        self.activation_list = []
        self.inhibition_list = []
        self.preprocess_network()

    def preprocess_network(self):
        """Preprocess network data, parse activation and inhibition lists"""
        for idx, row in self.network.iterrows():
            activated_by = (
                [int(x.strip()) - 1 for x in row["Activated by"].split(",")]
                if row["Activated by"] != "nothing"
                else []
            )
            inhibited_by = (
                [int(x.strip()) - 1 for x in row["Inhibited by"].split(",")]
                if row["Inhibited by"] != "nothing"
                else []
            )
            self.activation_list.append(activated_by)
            self.inhibition_list.append(inhibited_by)

    def system_equations(self, state, t, hill_coefficient):
        """Implement system differential equations"""
        dxdt = np.zeros(self.n_nodes)

        def hill_activation(x):
            """Hill activation function"""
            return (x**3) / (1 + x**3)

        def hill_inhibition(x):
            """Hill inhibition function"""
            return 1 / (1 + x**3)

        for idx in range(self.n_nodes):
            activated_by = self.activation_list[idx]
            inhibited_by = self.inhibition_list[idx]

            # Calculate activation term
            if activated_by:
                activation_values = [hill_activation(state[a]) for a in activated_by]
                activation = np.mean(activation_values)
            else:
                activation = 1.0

            # Calculate inhibition term
            if inhibited_by:
                inhibition_values = [hill_inhibition(state[i]) for i in inhibited_by]
                inhibition = np.prod(inhibition_values)
            else:
                inhibition = 1.0

            # Calculate the derivative of the node
            dxdt[idx] = activation * inhibition - state[idx]

        return dxdt

    def simulate_with_hill(self, hill_coefficient, initial_states):
        """Simulate system with given Hill coefficient"""
        t = np.linspace(0, 100, 1000)
        results = []

        for init_state in initial_states:
            solution = odeint(
                self.system_equations, init_state, t, args=(hill_coefficient,)
            )
            results.append(solution[-1])  # Take final steady state

        return np.array(results)

    def objective_function(self, hill_coefficient):
        """Objective function: Evaluate fitness of Hill coefficient"""
        score = 0

        # 1. Steady state score
        initial_states = self.generate_initial_states(30)  # Adjust based on network size
        final_states = self.simulate_with_hill(hill_coefficient, initial_states)
        unique_states = self.identify_unique_states(final_states)

        # Estimate expected number of attractors based on network size
        expected_attractor_count = max(2, self.n_nodes // 3)  # Simple heuristic estimate
        score += abs(len(unique_states) - expected_attractor_count)

        # 2. Switch characteristic score
        switch_score = self.evaluate_switch_behavior(hill_coefficient)
        score += switch_score

        # 3. Steady state distribution score
        if len(unique_states) > 0:
            state_distribution = np.std(
                [
                    len(np.where(np.all(np.abs(final_states - state) < 0.1, axis=1))[0])
                    for state in unique_states
                ]
            )
            score += state_distribution  # Penalize uneven distribution

        # 4. System stability score
        stability_score = np.mean(
            [
                np.std(trajectory)
                for trajectory in self.simulate_trajectories(hill_coefficient)
            ]
        )
        score += stability_score

        return score

    def evaluate_switch_behavior(self, hill_coefficient):
        """Evaluate system's switch characteristics"""
        x = np.linspace(0, 1, 100)
        hill_function = (hill_coefficient * x**3) / (1 + hill_coefficient * x**3)

        # Calculate slope in the 0.4-0.6 range
        mid_slope = np.mean(np.diff(hill_function[40:60]) / np.diff(x[40:60]))

        # Ideal slope is approximately 4
        return abs(mid_slope - 4)

    def identify_unique_states(self, states, tolerance=1e-2):
        """Identify unique steady states using clustering method"""
        if len(states) == 0:
            return []

        # Use DBSCAN clustering to find steady states
        clustering = DBSCAN(eps=tolerance, min_samples=2).fit(states)
        unique_labels = np.unique(clustering.labels_)

        # Take the mean of each cluster as the steady state
        unique_states = []
        for label in unique_labels:
            if label != -1:  # Ignore noise points
                cluster_states = states[clustering.labels_ == label]
                unique_states.append(np.mean(cluster_states, axis=0))

        return unique_states

    def generate_initial_states(self, n_states):
        """Generate random initial states"""
        return np.random.random((n_states, self.n_nodes))

    def optimize_hill_coefficient(self):
        """Optimize Hill coefficient"""
        result = minimize(
            self.objective_function,
            x0=8.0,  # Initial guess value
            bounds=[(self.min_hill, self.max_hill)],
            method="L-BFGS-B",
        )

        return {
            "optimal_hill_coefficient": result.x[0],
            "optimization_success": result.success,
            "objective_value": result.fun,
        }

    def analyze_hill_coefficient_range(self, start=1, end=20, step=1):
        """Analyze effects of different Hill coefficients"""
        results = []
        hill_range = np.arange(start, end + step, step)

        for hill in hill_range:
            score = self.objective_function(hill)
            results.append({"hill_coefficient": hill, "score": score})

        return pd.DataFrame(results)

    def plot_analysis_results(self, results_df):
        """Plot analysis results"""
        plt.figure(figsize=(10, 6))
        plt.plot(results_df["hill_coefficient"], results_df["score"], marker="o")
        plt.xlabel("Hill Coefficient")
        plt.ylabel("Objective Score")
        plt.title("Hill Coefficient Analysis")
        plt.grid(True)

    def simulate_trajectories(self, hill_coefficient, n_trajectories=5):
        """Simulate multiple trajectories to evaluate system stability"""
        t = np.linspace(0, 50, 500)  # Short time range
        initial_states = self.generate_initial_states(n_trajectories)
        trajectories = []

        for init_state in initial_states:
            solution = odeint(
                self.system_equations, init_state, t, args=(hill_coefficient,)
            )
            trajectories.append(solution)

        return trajectories


def main():
    # Add command line argument parsing
    parser = argparse.ArgumentParser(
        description="Optimize Hill coefficient for gene network"
    )
    parser.add_argument(
        "network_file", type=str, help="Path to the network structure file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/hill_analysis",
        help="Output directory for analysis results",
    )
    args = parser.parse_args()

    # Ensure input file exists
    network_path = Path(args.network_file)
    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Hill coefficient optimization...")
    # Create optimizer and run analysis
    optimizer = HillCoefficientOptimizer(network_path)

    print("Running optimization...")
    # Direct optimization
    optimal_result = optimizer.optimize_hill_coefficient()
    print(f"\nOptimization Results:")
    print(f"Optimal Hill coefficient: {optimal_result['optimal_hill_coefficient']:.2f}")
    print(f"Optimization success: {optimal_result['optimization_success']}")
    print(f"Final objective value: {optimal_result['objective_value']:.4f}")

    print("\nAnalyzing Hill coefficient range...")
    # Analyze effects of different Hill coefficients
    results = optimizer.analyze_hill_coefficient_range()

    # Save results
    results.to_csv(output_dir / "hill_coefficient_analysis.csv", index=False)
    optimizer.plot_analysis_results(results)
    plt.savefig(output_dir / "hill_coefficient_analysis.png")
    plt.close()

    # Save detailed report
    with open(output_dir / "optimization_report.txt", "w") as f:
        f.write("=== Hill Coefficient Optimization Report ===\n\n")
        f.write(f"Input network file: {network_path}\n")
        f.write(
            f"Optimal Hill coefficient: {optimal_result['optimal_hill_coefficient']:.2f}\n"
        )
        f.write(f"Optimization success: {optimal_result['optimization_success']}\n")
        f.write(f"Final objective value: {optimal_result['objective_value']:.4f}\n\n")

        f.write("Analysis across Hill coefficient range:\n")
        f.write(results.to_string())

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
