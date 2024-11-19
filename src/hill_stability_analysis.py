# -*- coding: utf-8 -*-
"""
@title Stability Analysis
@auther mengxu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from hill_coefficient_optimizer import HillCoefficientOptimizer
from tqdm import tqdm


class StabilityAnalyzer:
    def __init__(
        self, network_file, n_runs=50, output_dir="results/stability_analysis"
    ):
        """
        Initialize stability analyzer

        Args:
            network_file: Path to network file
            n_runs: Number of optimization runs
            output_dir: Directory to save results
        """
        self.network_file = network_file
        self.n_runs = n_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_analysis(self):
        """Run multiple optimizations and collect results"""
        print(f"Running {self.n_runs} optimizations...")

        for _ in tqdm(range(self.n_runs)):
            optimizer = HillCoefficientOptimizer(self.network_file)
            result = optimizer.optimize_hill_coefficient()
            self.results.append(
                {
                    "hill_coefficient": result["optimal_hill_coefficient"],
                    "objective_value": result["objective_value"],
                    "success": result["optimization_success"],
                }
            )

        return pd.DataFrame(self.results)

    def analyze_results(self):
        """Analyze optimization results and generate statistics"""
        df = pd.DataFrame(self.results)

        stats = {
            "hill_coefficient": {
                "mean": df["hill_coefficient"].mean(),
                "std": df["hill_coefficient"].std(),
                "min": df["hill_coefficient"].min(),
                "max": df["hill_coefficient"].max(),
                "median": df["hill_coefficient"].median(),
            },
            "objective_value": {
                "mean": df["objective_value"].mean(),
                "std": df["objective_value"].std(),
                "min": df["objective_value"].min(),
                "max": df["objective_value"].max(),
                "median": df["objective_value"].median(),
            },
            "success_rate": df["success"].mean() * 100,
        }

        return stats, df

    def plot_results(self, df):
        """Generate visualization plots for the results"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))

        # 1. Hill coefficient distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x="hill_coefficient", bins=20)
        plt.title("Distribution of Optimal Hill Coefficients")
        plt.xlabel("Hill Coefficient")
        plt.ylabel("Count")

        # 2. Objective value distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=df, x="objective_value", bins=20)
        plt.title("Distribution of Objective Values")
        plt.xlabel("Objective Value")
        plt.ylabel("Count")

        # 3. Box plots
        plt.subplot(2, 2, 3)
        df_melted = pd.melt(df[["hill_coefficient", "objective_value"]])
        sns.boxplot(data=df_melted, x="variable", y="value")
        plt.title("Box Plots of Results")
        plt.xticks(rotation=45)

        # 4. Scatter plot
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x="hill_coefficient", y="objective_value")
        plt.title("Hill Coefficient vs Objective Value")
        plt.xlabel("Hill Coefficient")
        plt.ylabel("Objective Value")

        plt.tight_layout()
        plt.savefig(self.output_dir / "stability_analysis.png")
        plt.close()

    def save_results(self, stats, df):
        """Save analysis results to files"""
        # Save raw data
        df.to_csv(self.output_dir / "optimization_runs.csv", index=False)

        # Save statistics
        with open(self.output_dir / "statistics.txt", "w") as f:
            f.write("=== Optimization Stability Analysis ===\n\n")

            f.write("Hill Coefficient Statistics:\n")
            for key, value in stats["hill_coefficient"].items():
                f.write(f"{key}: {value:.4f}\n")

            f.write("\nObjective Value Statistics:\n")
            for key, value in stats["objective_value"].items():
                f.write(f"{key}: {value:.4f}\n")

            f.write(f"\nOptimization Success Rate: {stats['success_rate']:.2f}%\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze optimization stability")
    parser.add_argument("network_file", type=str, help="Path to network file")
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of optimization runs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/stability_analysis",
        help="Output directory",
    )
    args = parser.parse_args()

    analyzer = StabilityAnalyzer(args.network_file, args.runs, args.output)

    # Run optimizations
    results_df = analyzer.run_analysis()

    # Analyze results
    stats, df = analyzer.analyze_results()

    # Generate plots
    analyzer.plot_results(df)

    # Save results
    analyzer.save_results(stats, df)

    print(f"\nAnalysis complete. Results saved to {args.output}")
    print("\nSummary Statistics:")
    print(
        f"Hill Coefficient: mean = {stats['hill_coefficient']['mean']:.2f} ± {stats['hill_coefficient']['std']:.2f}"
    )
    print(
        f"Objective Value: mean = {stats['objective_value']['mean']:.2f} ± {stats['objective_value']['std']:.2f}"
    )
    print(f"Success Rate: {stats['success_rate']:.1f}%")


if __name__ == "__main__":
    main()
