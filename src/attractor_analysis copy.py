# -*- coding: utf-8 -*-
"""
@title Attractor Analysis
@auther mengxu
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from scipy.cluster import hierarchy


class AttractorAnalyzer:
    def __init__(self, attractor_file, network_file=None, gene_mapping_file=None):
        self.attractor_file = attractor_file
        self.network_file = network_file
        self.data = self._load_data()
        self.total_runs = self.data["frequency"].sum()
        self.gene_names = self._load_gene_mapping(gene_mapping_file)

    def _load_data(self):
        data = []
        with open(self.attractor_file, "r") as f:
            for line in f:
                count, state = line.strip().split("\t")
                state = eval(state)
                data.append(
                    {"frequency": int(count), "state": np.array(state, dtype=float)}
                )
        return pd.DataFrame(data)

    def _load_gene_mapping(self, mapping_file):
        """Load gene-node mapping"""
        if mapping_file is None:
            return {i: f"Node {i+1}" for i in range(len(self.data.iloc[0]["state"]))}

        try:
            gene_map = {}
            with open(mapping_file, "r") as f:
                next(f)
                for line in f:
                    if line.strip():  # Ensure not an empty line
                        gene, number = line.strip().split("\t")
                        # Since the indices in the data start from 0, we subtract 1 here
                        gene_map[int(number) - 1] = gene
            return gene_map
        except Exception as e:
            print(f"Error loading gene mapping: {e}")
            return {i: f"Node {i+1}" for i in range(len(self.data.iloc[0]["state"]))}

    def get_gene_name(self, node_index):
        """Get gene name for a node"""
        return self.gene_names.get(node_index, f"Node {node_index+1}")

    def plot_all_attractors(self, save_path=None):
        """Plot heatmap of all attractors"""
        n_attractors = len(self.data)
        fig, axes = plt.subplots(n_attractors, 1, figsize=(12, 2 * n_attractors))
        if n_attractors == 1:
            axes = [axes]

        sorted_data = self.data.sort_values("frequency", ascending=False)

        for idx, (_, row) in enumerate(sorted_data.iterrows()):
            percentage = (row["frequency"] / self.total_runs) * 100
            sns.heatmap(
                [row["state"]],
                ax=axes[idx],
                cmap="RdYlBu_r",
                xticklabels=[self.get_gene_name(i) for i in range(len(row["state"]))],
                yticklabels=[f"Freq: {row['frequency']} ({percentage:.1f}%)"],
            )
            axes[idx].set_title(f"Attractor {idx+1}")
            axes[idx].set_xticklabels(
                axes[idx].get_xticklabels(), rotation=45, ha="right"
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_correlation_matrix(self, save_path=None):
        """Plot correlation matrix between genes"""
        states = np.vstack([row["state"] for _, row in self.data.iterrows()])
        corr_matrix = np.corrcoef(states.T)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            xticklabels=[self.get_gene_name(i) for i in range(len(corr_matrix))],
            yticklabels=[self.get_gene_name(i) for i in range(len(corr_matrix))],
            cmap="RdYlBu_r",
            center=0,
            annot=True,
            fmt=".2f",
        )

        plt.title("Gene Correlation Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_hierarchical_clustering(self, save_path=None):
        """Generate hierarchical clustering heatmap of attractor states"""
        states = np.vstack([row["state"] for _, row in self.data.iterrows()])
        frequencies = self.data["frequency"].values

        g = sns.clustermap(
            states,
            cmap="RdYlBu_r",
            annot=True,
            fmt=".2f",
            xticklabels=[self.get_gene_name(i) for i in range(states.shape[1])],
            yticklabels=[f"A{i+1} ({freq})" for i, freq in enumerate(frequencies)],
            figsize=(12, 8),
            dendrogram_ratio=0.15,
            cbar_pos=(0.02, 0.8, 0.03, 0.2),
        )

        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        g.fig.suptitle("Hierarchical Clustering of Attractors", y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_state_network(self, save_path=None):
        """Plot attractor state network"""
        G = nx.Graph()

        for idx, row in self.data.iterrows():
            G.add_node(
                idx,
                state=row["state"],
                frequency=row["frequency"],
                size=1000 * row["frequency"] / self.total_runs,
            )

        for i in G.nodes():
            for j in G.nodes():
                if i < j:
                    similarity = 1 - np.mean(
                        np.abs(G.nodes[i]["state"] - G.nodes[j]["state"])
                    )
                    if similarity > 0.7:  # Threshold adjustable
                        G.add_edge(i, j, weight=similarity)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=[G.nodes[node]["size"] for node in G.nodes()],
            node_color="lightblue",
            alpha=0.7,
        )

        nx.draw_networkx_edges(G, pos, alpha=0.2)

        labels = {i: f"A{i+1}\n({G.nodes[i]['frequency']})" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels)

        plt.title("Attractor State Network")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def analyze_gene_importance(self):
        """分析基因重要性"""
        states = np.vstack([row["state"] for _, row in self.data.iterrows()])
        frequencies = self.data["frequency"].values

        # Calculate various importance metrics
        importance_metrics = {}
        n_genes = states.shape[1]

        # 1. Activity frequency (average expression level across all attractors, weighted by frequency)
        weighted_states = states * frequencies[:, np.newaxis]
        activity_level = weighted_states.sum(axis=0) / frequencies.sum()

        # 2. Variation (standard deviation of expression levels)
        variation = np.std(states, axis=0)

        # 3. Connectivity (sum of absolute correlation values with other genes)
        corr_matrix = np.abs(np.corrcoef(states.T))
        connectivity = np.sum(corr_matrix, axis=0) - 1  # Subtract self-correlation

        # 4. Discrimination of attractors (entropy)
        def calculate_entropy(gene_states):
            hist, _ = np.histogram(gene_states, bins=10, range=(0, 1))
            prob = hist / hist.sum()
            prob = prob[prob > 0]  # Remove zero probability
            return -np.sum(prob * np.log2(prob))

        entropy = np.array([calculate_entropy(states[:, i]) for i in range(n_genes)])

        # Integrate all metrics
        for i in range(n_genes):
            gene_name = self.get_gene_name(i)
            importance_metrics[gene_name] = {
                "activity_level": activity_level[i],
                "variation": variation[i],
                "connectivity": connectivity[i],
                "entropy": entropy[i],
                # Overall score (normalized average)
                "overall_score": 0,  # Will be calculated below
            }

        # Normalize and calculate overall scores
        metrics_array = np.array(
            [
                [
                    v[key]
                    for key in [
                        "activity_level",
                        "variation",
                        "connectivity",
                        "entropy",
                    ]
                ]
                for v in importance_metrics.values()
            ]
        )
        normalized_metrics = (metrics_array - metrics_array.min(axis=0)) / (
            metrics_array.max(axis=0) - metrics_array.min(axis=0)
        )
        overall_scores = np.mean(normalized_metrics, axis=1)

        for i, gene_name in enumerate(importance_metrics.keys()):
            importance_metrics[gene_name]["overall_score"] = overall_scores[i]

        return importance_metrics

    def plot_gene_importance(self, importance_metrics, save_path=None):
        """Plot gene importance analysis results"""
        # Sort by overall score
        sorted_genes = sorted(
            importance_metrics.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True,
        )

        # Prepare plotting data
        genes = [g[0] for g in sorted_genes]
        metrics = np.array(
            [
                [
                    g[1][key]
                    for key in [
                        "activity_level",
                        "variation",
                        "connectivity",
                        "entropy",
                    ]
                ]
                for g in sorted_genes
            ]
        )

        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            metrics.T,
            xticklabels=genes,
            yticklabels=["Activity Level", "Variation", "Connectivity", "Entropy"],
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
        )

        plt.title("Gene Importance Analysis")
        plt.xticks(rotation=45, ha="right")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return sorted_genes

    def plot_gene_ranking(self, importance_metrics, save_path=None):
        """Plot horizontal bar chart of gene importance ranking (descending order)"""
        # Extract overall scores and sort
        gene_scores = {
            gene: metrics["overall_score"]
            for gene, metrics in importance_metrics.items()
        }
        sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)

        # Prepare plotting data
        genes = [x[0] for x in sorted_genes]
        scores = [x[1] for x in sorted_genes]

        # Create figure, reduce height
        base_size = 6  # Base width
        height_ratio = 1.5  # Reduce height ratio (was 2)
        plt.figure(figsize=(base_size, base_size * height_ratio))

        # Reverse gene order so highest score is at the top
        genes = genes[::-1]
        scores = scores[::-1]

        # Plot horizontal bar chart
        bars = plt.barh(range(len(genes)), scores, color="steelblue")

        # Set y-axis labels (in descending order)
        plt.yticks(range(len(genes)), genes)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(
                width + 0.01, i, f"{width:.3f}", ha="left", va="center", fontsize=10
            )

        # Set title and labels
        plt.title("Gene Importance Ranking", pad=20, fontsize=12)
        plt.xlabel("Overall Score", fontsize=11)

        # Add grid lines
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_phase_portrait(self, node1=0, node2=1, save_path=None):
        """Plot enhanced phase portrait of two nodes with better attractor visualization"""
        # Create grid points
        x = np.linspace(0, 1.0, 30)
        y = np.linspace(0, 1.0, 30)
        X, Y = np.meshgrid(x, y)

        # Get attractor states and frequencies
        attractors = np.array([row["state"] for _, row in self.data.iterrows()])
        frequencies = np.array([row["frequency"] for _, row in self.data.iterrows()])

        # Calculate vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Simple vector field calculation
        for i in range(len(x)):
            for j in range(len(y)):
                point = np.array([X[i, j], Y[i, j]])
                distances = [
                    np.linalg.norm(point - attractor[[node1, node2]])
                    for attractor in attractors
                ]
                nearest_attractor = attractors[np.argmin(distances)][[node1, node2]]

                U[i, j] = nearest_attractor[0] - X[i, j]
                V[i, j] = nearest_attractor[1] - Y[i, j]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(9, 8))

        # Plot streamlines
        magnitude = np.sqrt(U**2 + V**2)
        strm = ax.streamplot(
            X,
            Y,
            U,
            V,
            color=magnitude,
            cmap="viridis",
            density=2,
            linewidth=1.5,
            arrowsize=1,
        )

        # Calculate normalized frequencies for point sizes
        max_size = 500
        min_size = 100
        normalized_frequencies = (frequencies - frequencies.min()) / (
            frequencies.max() - frequencies.min()
        )
        point_sizes = min_size + normalized_frequencies * (max_size - min_size)

        # Plot attractor points
        scatter_points = []
        for i, attractor in enumerate(attractors):
            scatter = ax.scatter(
                attractor[node1],
                attractor[node2],
                s=point_sizes[i],
                c=f"C{i}",
                alpha=0.7,
                label=f"A{i+1} ({frequencies[i]})",
            )
            scatter_points.append((attractor[node1], attractor[node2], f"A{i+1}"))

        # Add annotations with leader lines
        for x, y, label in scatter_points:
            # Find nearby points
            nearby_points = [
                (px, py)
                for px, py, _ in scatter_points
                if abs(px - x) < 0.1 and abs(py - y) < 0.1 and (px, py) != (x, y)
            ]

            # Adjust annotation position based on nearby points
            if nearby_points:
                xytext = (x + 0.15, y + 0.15)
            else:
                xytext = (x + 0.05, y + 0.05)

            ax.annotate(
                label,
                xy=(x, y),
                xytext=xytext,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                arrowprops=dict(
                    arrowstyle="->", connectionstyle="arc3,rad=0.2", alpha=0.7
                ),
            )

        # Customize plot
        ax.set_xlabel(f"{self.get_gene_name(node1)} State", fontsize=12)
        ax.set_ylabel(f"{self.get_gene_name(node2)} State", fontsize=12)
        ax.set_title("Phase Portrait with Attractor Basins", fontsize=14, pad=20)

        # Add colorbar
        fig.colorbar(strm.lines, label="Vector Field Magnitude")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.2)

        # Adjust layout
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def parse_differential_equations(self, network_file):
        """Parse and format differential equations"""
        if not network_file:
            return "No network file provided, cannot generate differential equations."

        frame = pd.read_csv(network_file, sep="\t")
        equations = []

        for idx, row in frame.iterrows():
            node_id = row["Node"]
            activated_by = str(row["Activated by"])
            inhibited_by = str(row["Inhibited by"])

            # Build activation term
            if activated_by != "nothing":
                if "," in activated_by:
                    activators = activated_by.split(",")
                    activation = "+".join(
                        [f"{self.get_gene_name(int(a.strip())-1)}³" for a in activators]
                    )
                else:
                    activation = f"{self.get_gene_name(int(activated_by.strip())-1)}³"
                activation = f"8({activation})/(1 + 8({activation}))"
            else:
                activation = "1"

            # Build inhibition term
            if inhibited_by != "nothing":
                if "," in inhibited_by:
                    inhibitors = inhibited_by.split(",")
                    inhibition = "*".join(
                        [
                            f"1/(1 + 8({self.get_gene_name(int(i.strip())-1)}³))"
                            for i in inhibitors
                        ]
                    )
                else:
                    inhibition = (
                        f"1/(1 + 8({self.get_gene_name(int(inhibited_by.strip())-1)}³))"
                    )
            else:
                inhibition = "1"

            # Build complete equation
            gene_name = self.get_gene_name(node_id - 1)
            equation = f"d{gene_name}/dt = {activation} * {inhibition} - {gene_name}"
            equations.append(equation)

        return equations

    def generate_comprehensive_report(self, output_dir):
        """Generate comprehensive analysis report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        self.plot_correlation_matrix(save_path=output_dir / "correlation_matrix.png")
        self.plot_hierarchical_clustering(
            save_path=output_dir / "hierarchical_clustering.png"
        )
        self.plot_state_network(save_path=output_dir / "state_network.png")

        # Analyze gene importance
        importance_metrics = self.analyze_gene_importance()
        sorted_genes = self.plot_gene_importance(
            importance_metrics, save_path=output_dir / "gene_importance.png"
        )
        # Add gene ranking plot
        self.plot_gene_ranking(
            importance_metrics, save_path=output_dir / "gene_ranking.png"
        )

        # Add phase portrait
        # Select top two genes to plot phase portrait
        top_genes = sorted(
            importance_metrics.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True,
        )[:2]
        node1 = list(self.gene_names.keys())[
            list(self.gene_names.values()).index(top_genes[0][0])
        ]
        node2 = list(self.gene_names.keys())[
            list(self.gene_names.values()).index(top_genes[1][0])
        ]
        self.plot_phase_portrait(
            node1, node2, save_path=output_dir / "phase_portrait_2d.png"
        )
        self.plot_phase_portrait_3d(
            node1, node2, save_path=output_dir / "phase_portrait_3d.png"
        )

        # Add three-gene phase portrait
        # Select top three genes based on importance
        top_genes = sorted(
            importance_metrics.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True,
        )[:3]
        node1 = list(self.gene_names.keys())[
            list(self.gene_names.values()).index(top_genes[0][0])
        ]
        node2 = list(self.gene_names.keys())[
            list(self.gene_names.values()).index(top_genes[1][0])
        ]
        node3 = list(self.gene_names.keys())[
            list(self.gene_names.values()).index(top_genes[2][0])
        ]

        self.plot_three_gene_portrait(
            node1, node2, node3, save_path=output_dir / "phase_portrait_three_genes.png"
        )

        # Generate report text
        with open(output_dir / "comprehensive_report.txt", "w") as f:
            f.write("=== Attractor Comprehensive Analysis Report ===\n\n")

            # Add differential equations section
            f.write("Network Differential Equation System:\n")
            equations = self.parse_differential_equations(self.network_file)
            if isinstance(equations, list):
                for eq in equations:
                    f.write(f"{eq}\n")
            else:
                f.write(equations + "\n")
            f.write("\n")

            # Basic statistics
            f.write(f"Total simulation runs: {self.total_runs}\n")
            f.write(f"Unique attractors: {len(self.data)}\n\n")

            # Gene importance ranking
            f.write("Gene importance ranking:\n")
            for gene_name, _ in sorted_genes:
                score = importance_metrics[gene_name]["overall_score"]
                f.write(f"{gene_name}: {score:.3f}\n")
            f.write("\n")

            # Attractor detailed information
            f.write("Attractor detailed information:\n")
            sorted_data = self.data.sort_values("frequency", ascending=False)
            for idx, row in sorted_data.iterrows():
                percentage = (row["frequency"] / self.total_runs) * 100
                f.write(f"\nAttractor {idx+1}:\n")
                f.write(f"Frequency: {row['frequency']} ({percentage:.2f}%)\n")
                f.write("Active genes: ")
                active_genes = [
                    self.get_gene_name(i) for i, v in enumerate(row["state"]) if v > 0.1
                ]
                f.write(", ".join(active_genes) if active_genes else "No active genes")
                f.write("\n")
                f.write(f"Full state: {row['state']}\n")

    def plot_phase_portrait_3d(self, node1=0, node2=1, save_path=None):
        """Plot 3D phase portrait with attractors and vector field

        Args:
            node1 (int): Index of the first node
            node2 (int): Index of the second node
            save_path (str, optional): Path to save the figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.colors import LinearSegmentedColormap

        # Create grid points
        x = np.linspace(0, 1.2, 20)  # Reduced grid density for 3D
        y = np.linspace(0, 1.2, 20)
        X, Y = np.meshgrid(x, y)

        # Get attractor states and frequencies
        attractors = np.array([row["state"] for _, row in self.data.iterrows()])
        frequencies = np.array([row["frequency"] for _, row in self.data.iterrows()])

        # Calculate vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        Z = np.zeros_like(X)  # Height field for vector field visualization

        # Calculate vector field and potential field
        for i in range(len(x)):
            for j in range(len(y)):
                point = np.array([X[i, j], Y[i, j]])
                # Calculate distances to all attractors
                distances = [
                    np.linalg.norm(point - attractor[[node1, node2]])
                    for attractor in attractors
                ]
                weights = np.exp(-np.array(distances) * 5)  # Exponential weighting
                weights = weights / weights.sum()  # Normalize weights

                # Calculate weighted influence of attractors
                nearest_attractor = attractors[np.argmin(distances)][[node1, node2]]
                U[i, j] = nearest_attractor[0] - X[i, j]
                V[i, j] = nearest_attractor[1] - Y[i, j]

                # Calculate height based on weighted sum of attractor influences
                Z[i, j] = np.sum(weights * frequencies) / frequencies.max()

        # Create figure
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection="3d")

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)

        # Plot vector field as quiver
        # Subsample the grid for clearer visualization
        skip = 2
        ax.quiver(
            X[::skip, ::skip],
            Y[::skip, ::skip],
            Z[::skip, ::skip],
            U[::skip, ::skip],
            V[::skip, ::skip],
            np.zeros_like(Z[::skip, ::skip]),
            length=0.1,
            normalize=True,
            color="gray",
            alpha=0.3,
        )

        # Plot attractors as scatter points
        normalized_frequencies = frequencies / frequencies.max()
        scatter_points = []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(attractors)))

        for i, (attractor, freq, color) in enumerate(
            zip(attractors, normalized_frequencies, colors)
        ):
            x, y = attractor[node1], attractor[node2]
            z = freq
            scatter = ax.scatter(
                [x],
                [y],
                [z],
                s=300 * freq,  # Size proportional to frequency
                c=[color],
                alpha=0.8,
                label=f"A{i+1} ({int(frequencies[i])})",
            )
            scatter_points.append((x, y, z, f"A{i+1}"))

        # Add annotations for attractors
        for x, y, z, label in scatter_points:
            ax.text(
                x + 0.05,
                y + 0.05,
                z + 0.05,
                label,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        # Customize plot
        ax.set_xlabel(f"{self.get_gene_name(node1)} State")
        ax.set_ylabel(f"{self.get_gene_name(node2)} State")
        ax.set_zlabel("Normalized Frequency")
        ax.set_title("3D Phase Portrait with Attractor Basins")

        # Add colorbar
        fig.colorbar(surf, ax=ax, label="Potential Field")

        # Adjust view angle for better visualization
        ax.view_init(elev=25, azim=45)

        # Add legend
        ax.legend()

        # Adjust layout
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_three_gene_portrait(self, node1=0, node2=1, node3=2, save_path=None):
        """Plot 3D phase portrait for three genes with consistent style as 2D version"""
        from mpl_toolkits.mplot3d import Axes3D

        # Create figure with proper size and layout
        fig = plt.figure(figsize=(13, 10))
        gs = fig.add_gridspec(
            1, 2, width_ratios=[20, 1]
        )  # Create grid for main plot and colorbar
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        cax = fig.add_subplot(gs[0, 1])  # Colorbar axis

        # Get attractor states and frequencies
        attractors = np.array([row["state"] for _, row in self.data.iterrows()])
        frequencies = np.array([row["frequency"] for _, row in self.data.iterrows()])

        # Create grid points with reduced density for 3D
        x = np.linspace(0, 1.2, 8)
        y = np.linspace(0, 1.2, 8)
        z = np.linspace(0, 1.2, 8)
        X, Y, Z = np.meshgrid(x, y, z)

        # Calculate vector field
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        W = np.zeros_like(Z)

        # Calculate vector field
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    point = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                    distances = [
                        np.linalg.norm(point - attractor[[node1, node2, node3]])
                        for attractor in attractors
                    ]
                    nearest_attractor = attractors[np.argmin(distances)][
                        [node1, node2, node3]
                    ]

                    U[i, j, k] = nearest_attractor[0] - X[i, j, k]
                    V[i, j, k] = nearest_attractor[1] - Y[i, j, k]
                    W[i, j, k] = nearest_attractor[2] - Z[i, j, k]

        # Calculate magnitude for coloring
        magnitude = np.sqrt(U**2 + V**2 + W**2)

        # Plot vector field using quiver
        skip = 2
        quiver = ax.quiver(
            X[::skip, ::skip, ::skip],
            Y[::skip, ::skip, ::skip],
            Z[::skip, ::skip, ::skip],
            U[::skip, ::skip, ::skip],
            V[::skip, ::skip, ::skip],
            W[::skip, ::skip, ::skip],
            length=0.1,
            normalize=True,
            color="gray",
            alpha=0.3,
        )

        # Calculate normalized frequencies for point sizes
        max_size = 500
        min_size = 100
        normalized_frequencies = (frequencies - frequencies.min()) / (
            frequencies.max() - frequencies.min()
        )
        point_sizes = min_size + normalized_frequencies * (max_size - min_size)

        # Plot attractors
        scatter_points = []
        for i, attractor in enumerate(attractors):
            scatter = ax.scatter(
                attractor[node1],
                attractor[node2],
                attractor[node3],
                s=point_sizes[i],
                c=f"C{i}",
                alpha=0.7,
                label=f"A{i+1} ({frequencies[i]})",
            )
            scatter_points.append(
                (attractor[node1], attractor[node2], attractor[node3], f"A{i+1}")
            )

        # Add annotations with leader lines
        for x, y, z, label in scatter_points:
            # Find nearby points
            nearby_points = [
                (px, py, pz)
                for px, py, pz, _ in scatter_points
                if abs(px - x) < 0.1
                and abs(py - y) < 0.1
                and abs(pz - z) < 0.1
                and (px, py, pz) != (x, y, z)
            ]

            # Adjust annotation position based on nearby points
            if nearby_points:
                xytext = (x + 0.15, y + 0.15, z + 0.15)
            else:
                xytext = (x + 0.05, y + 0.05, z + 0.05)

            ax.text(
                xytext[0],
                xytext[1],
                xytext[2],
                label,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        # Add some example trajectories
        n_trajectories = 8
        for _ in range(n_trajectories):
            # Random starting point
            start = np.random.rand(3) * 1.2
            trajectory = [start]

            # Generate trajectory
            for _ in range(20):
                point = trajectory[-1]
                distances = [
                    np.linalg.norm(point - attractor[[node1, node2, node3]])
                    for attractor in attractors
                ]
                nearest_attractor = attractors[np.argmin(distances)][
                    [node1, node2, node3]
                ]

                # Move towards nearest attractor
                direction = nearest_attractor - point
                next_point = point + direction * 0.1
                trajectory.append(next_point)

            # Plot trajectory
            trajectory = np.array(trajectory)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                "--",
                color="gray",
                alpha=0.3,
            )

        # Customize plot
        ax.set_xlabel(f"{self.get_gene_name(node1)} State", fontsize=12)
        ax.set_ylabel(f"{self.get_gene_name(node2)} State", fontsize=12)
        ax.set_zlabel(f"{self.get_gene_name(node3)} State", fontsize=12)
        ax.set_title("Three-Gene Phase Portrait", fontsize=14, pad=20)

        # Add colorbar for vector field magnitude
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])  # Dummy array for the colorbar

        # Create space for colorbar and add it
        fig.colorbar(sm, cax=cax, label="Vector Field Magnitude")

        # Set view angle
        ax.view_init(elev=20, azim=45)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.2)

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.05)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze network attractors")
    parser.add_argument("-i", "--input", required=True, help="Input attractor file")
    parser.add_argument("-n", "--network", help="Input network file")
    parser.add_argument("-g", "--gene_mapping", help="Gene mapping file")
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for analysis"
    )
    args = parser.parse_args()

    analyzer = AttractorAnalyzer(args.input, args.network, args.gene_mapping)
    analyzer.generate_comprehensive_report(args.output)


if __name__ == "__main__":
    main()
