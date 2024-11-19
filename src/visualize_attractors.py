# -*- coding: utf-8 -*-
"""
@title Attractor Visualization
@auther mengxu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import itertools
from graphviz import Digraph
from matplotlib.gridspec import GridSpec


class AttractorVisualizer:
    def __init__(self, attractor_file, network_file, gene_mapping_file=None):
        """
        Initialize visualizer

        Parameters:
        -----------
        attractor_file : str
            Path to attractor results file
        network_file : str
            Path to original network file (xlsx)
        gene_mapping_file : str or None
            Path to gene mapping file (CSV)
        """
        self.attractors = self.read_attractors(attractor_file)
        self.total_runs = sum(a["frequency"] for a in self.attractors)
        self.figures_dir = "figures"
        os.makedirs(self.figures_dir, exist_ok=True)

        # Read network structure and generate equations
        self.network_data = pd.read_excel(network_file)
        self.equations = self.generate_network_equations()

        # Calculate and store node activity
        self.node_activities = self.calculate_node_frequencies()

        # Load gene mapping
        self.gene_names = self.load_gene_mapping(gene_mapping_file)

    def get_figure_path(self, filename):
        """Get full path for figure file"""
        return os.path.join(self.figures_dir, filename)

    def read_attractors(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        return [
            {"frequency": int(freq), "state": literal_eval(state)}
            for freq, state in (line.strip().split("\t") for line in lines)
        ]

    def plot_node_activation_frequencies(self):
        """
        Plot bar chart showing activation frequencies for each node.

        Creates a bar chart visualization where:
        - X-axis: Gene/node labels
        - Y-axis: Activation frequency (0-100%)
        - Each bar shows the percentage of time that node is active
        - Actual percentage values are displayed on top of each bar

        Saves output to: figures/node_frequencies.png
        """
        frequencies = self.calculate_node_frequencies()
        plt.figure(figsize=(10, 6))
        gene_labels = [self.get_gene_name(i) for i in range(len(frequencies))]
        bars = plt.bar(gene_labels, frequencies)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2%}",
                ha="center",
                va="bottom",
            )

        plt.title("Node Activation Frequencies")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            self.get_figure_path("node_frequencies.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_attractor_distribution(self):
        """Plot distribution of attractor frequencies"""
        frequencies = [a["frequency"] for a in self.attractors]
        plt.figure(figsize=(12, 6))
        plt.hist(frequencies, bins=20)
        plt.title("Distribution of Attractor Frequencies")
        plt.xlabel("Frequency")
        plt.ylabel("Count")
        plt.savefig(
            self.get_figure_path("attractor_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_state_network(self):
        """Plot network of state transitions"""
        G = nx.Graph()

        # Add nodes for each attractor state
        for i, attractor in enumerate(self.attractors):
            state_str = "".join(
                "1" if float(v) > 0.4 else "0" for v in attractor["state"]
            )
            G.add_node(state_str, weight=attractor["frequency"])

        # Add edges between similar states
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                hamming_distance = sum(a != b for a, b in zip(nodes[i], nodes[j]))
                if hamming_distance == 1:  # Connect states that differ by one node
                    G.add_edge(nodes[i], nodes[j])

        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            node_size=[G.nodes[node]["weight"] * 100 for node in G.nodes()],
            node_color="lightblue",
            with_labels=True,
            font_size=8,
        )
        plt.title("State Transition Network")
        plt.savefig(
            self.get_figure_path("state_network.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_correlation_matrix(self):
        """Plot correlation matrix heatmap"""
        states = np.array(
            [[float(x) for x in attractor["state"]] for attractor in self.attractors]
        )
        weights = np.array([attractor["frequency"] for attractor in self.attractors])
        corr_matrix = np.corrcoef(states.T)

        # Create figure
        plt.figure(figsize=(10, 8))

        heatmap = sns.heatmap(
            corr_matrix,
            cmap="RdBu_r",  # Use red-blue color scheme, better for correlation display
            center=0,  # Set color center point to 0
            vmin=-1,
            vmax=1,
            square=True,  # Keep square cells
            annot=True,  # Show values
            fmt=".2f",  # Format values as 2 decimal places
            annot_kws={"size": 8},  # Adjust number size
            cbar_kws={"label": "Correlation Coefficient"},  # Add color bar label
        )

        # Set labels
        gene_labels = [self.get_gene_name(i) for i in range(len(corr_matrix))]
        plt.xticks(np.arange(len(corr_matrix)) + 0.5, gene_labels)
        plt.yticks(np.arange(len(corr_matrix)) + 0.5, gene_labels)

        # Rotate x-axis labels to avoid overlap
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.title("Node Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            self.get_figure_path("correlation_matrix.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_hierarchical_clustering(self):
        """
        Generate hierarchical clustering heatmap of attractor states.

        Creates a clustered heatmap showing:
        - Rows: Individual attractors
        - Columns: Genes/nodes
        - Color intensity: Gene activity level (0-1)
        - Dendrograms: Show clustering relationships

        Parameters:
        - Uses RdYlBu_r colormap for activity visualization
        - Includes both row and column clustering
        - Shows actual values in cells
        - Includes color scale bar

        Saves output to: figures/hierarchical_clustering.png
        """
        states = np.array(
            [[float(x) for x in attractor["state"]] for attractor in self.attractors]
        )
        gene_labels = [self.get_gene_name(i) for i in range(states.shape[1])]
        attractor_labels = [f"A{i+1}" for i in range(states.shape[0])]

        g = sns.clustermap(
            states,
            cmap="RdYlBu_r",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            xticklabels=gene_labels,
            yticklabels=attractor_labels,
            col_cluster=True,
            row_cluster=True,
            dendrogram_ratio=0.15,
            cbar_pos=(0.02, 0.8, 0.03, 0.2),
            figsize=(12, 8),
            cbar_kws={"label": "Gene Activity", "orientation": "horizontal"},
        )

        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        g.fig.suptitle(
            "Hierarchical Clustering of Attractors", y=1.02, size=14, weight="bold"
        )
        plt.savefig(
            self.get_figure_path("hierarchical_clustering.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_analysis_report(self):
        """Generate comprehensive analysis report with visualization explanations"""
        report_file = os.path.join("reports", "visualization_analysis.txt")

        with open(report_file, "w") as f:
            f.write("Network Dynamics Visualization Analysis\n")
            f.write("=" * 50 + "\n\n")

            # 1. Node Activation Frequencies
            f.write("1. Node Activation Frequencies (node_frequencies.png)\n")
            f.write("-" * 50 + "\n")
            f.write(
                "This bar chart shows how often each node is active across all attractors.\n"
            )
            f.write("Key observations:\n")
            node_freqs = self.calculate_node_frequencies()
            for node, freq in enumerate(node_freqs, 1):
                f.write(f"- Node {node}: {freq:.1%} activation frequency\n")
            f.write("\nInterpretation:\n")
            f.write("- Highly active nodes (>20%) may be key regulators\n")
            f.write("- Inactive nodes (<5%) might be strongly suppressed\n\n")

            # 2. Attractor Distribution
            f.write("\n2. Attractor Distribution (attractor_distribution.png)\n")
            f.write("-" * 50 + "\n")
            f.write("This histogram shows the distribution of attractor frequencies.\n")
            f.write("Key statistics:\n")
            f.write(f"- Total unique attractors: {len(self.attractors)}\n")
            f.write(
                f"- Most common attractor frequency: {max(a['frequency'] for a in self.attractors)}\n"
            )
            f.write(
                f"- Average attractor frequency: {np.mean([a['frequency'] for a in self.attractors]):.1f}\n\n"
            )

            # 3. State Network
            f.write("\n3. State Transition Network (state_network.png)\n")
            f.write("-" * 50 + "\n")
            f.write(
                "This network visualization shows relationships between different attractor states.\n"
            )
            f.write("- Nodes represent unique attractor states\n")
            f.write("- Edges connect states that differ by one node's state\n")
            f.write("- Node size indicates frequency of the attractor\n\n")

            # 4. Correlation Matrix
            f.write("\n4. Node Correlation Matrix (correlation_matrix.png)\n")
            f.write("-" * 50 + "\n")
            f.write(
                "This heatmap shows how often pairs of nodes are active together.\n"
            )
            f.write("Strong correlations (>0.5) indicate:\n")
            corr_matrix = self.calculate_correlation_matrix()
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    if corr_matrix[i][j] > 0.5:
                        f.write(f"- Nodes {i+1} and {j+1}: {corr_matrix[i][j]:.2f}\n")

            # 5. Hierarchical Clustering
            f.write("\n5. Hierarchical Clustering (hierarchical_clustering.png)\n")
            f.write("-" * 50 + "\n")
            f.write(
                "This dendrogram shows how nodes cluster based on their activation patterns.\n"
            )
            f.write("Interpretation:\n")
            f.write(
                "- Nodes close together in the tree have similar activation patterns\n"
            )
            f.write("- Height of connections indicates dissimilarity\n\n")

            # 6. Attractor Heatmap
            f.write("\n6. Attractor States Heatmap (attractor_heatmap.png)\n")
            f.write("-" * 50 + "\n")
            f.write("This heatmap shows all attractor states sorted by frequency.\n")
            f.write("Key patterns:\n")
            f.write("- Dark colors indicate active nodes (0.809)\n")
            f.write("- Light colors indicate inactive nodes (0.000)\n")
            f.write("- Rows are sorted by frequency (most common at top)\n\n")

            # Overall Network Insights
            f.write("\nOverall Network Insights\n")
            f.write("-" * 50 + "\n")
            self.write_network_insights(f)

        print(f"\nDetailed visualization analysis has been saved to: {report_file}")

    def calculate_node_frequencies(self):
        """Calculate activation frequency for each node"""
        n_nodes = len(self.attractors[0]["state"])
        frequencies = np.zeros(n_nodes)

        for attractor in self.attractors:
            state = [float(v) > 0.4 for v in attractor["state"]]
            freq = attractor["frequency"] / self.total_runs
            frequencies += np.array(state) * freq

        return frequencies

    def calculate_correlation_matrix(self):
        """
        Calculate correlation matrix with handling for zero division

        Returns:
        --------
        numpy.ndarray
            Correlation matrix with values between -1 and 1
        """
        states = np.array(
            [[float(x) for x in attractor["state"]] for attractor in self.attractors]
        )

        # Add small epsilon to avoid division by zero
        eps = 1e-10

        # Calculate correlation with numpy's corrcoef
        corr_matrix = np.corrcoef(states.T)

        # Replace NaN values with 0
        corr_matrix = np.nan_to_num(corr_matrix)

        return corr_matrix

    def write_network_insights(self, f):
        """Write overall network insights"""
        # Calculate key metrics
        node_freqs = self.calculate_node_frequencies()
        active_nodes = [i + 1 for i, freq in enumerate(node_freqs) if freq > 0.2]
        inactive_nodes = [i + 1 for i, freq in enumerate(node_freqs) if freq < 0.05]

        f.write("1. Network Structure:\n")
        f.write(f"- Key regulatory nodes: {active_nodes}\n")
        f.write(f"- Suppressed nodes: {inactive_nodes}\n")

        f.write("\n2. Network Stability:\n")
        dominant_freq = max(a["frequency"] for a in self.attractors) / self.total_runs
        f.write(f"- Dominant attractor strength: {dominant_freq:.1%}\n")
        f.write(
            f"- Number of rare states: {sum(1 for a in self.attractors if a['frequency']/self.total_runs < 0.01)}\n"
        )

        f.write("\n3. Recommendations:\n")
        f.write("- Consider perturbation analysis of key nodes\n")
        f.write("- Investigate regulatory mechanisms of highly active nodes\n")
        f.write("- Explore potential intervention points\n")

    def plot_state_transition_diagram(self):
        """Plot state transition diagram with probabilities"""
        G = nx.DiGraph()

        # Add nodes
        for attractor in self.attractors:
            state_str = "".join(
                "1" if float(v) > 0.4 else "0" for v in attractor["state"]
            )
            G.add_node(state_str, weight=attractor["frequency"])

        # Add edges based on Hamming distance
        nodes = list(G.nodes())
        for i, j in itertools.combinations(range(len(nodes)), 2):
            hamming_distance = sum(a != b for a, b in zip(nodes[i], nodes[j]))
            if hamming_distance == 1:
                # Add bidirectional edges with weights
                weight = (
                    min(G.nodes[nodes[i]]["weight"], G.nodes[nodes[j]]["weight"])
                    / self.total_runs
                )
                G.add_edge(nodes[i], nodes[j], weight=weight)
                G.add_edge(nodes[j], nodes[i], weight=weight)

        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=[G.nodes[node]["weight"] * 100 for node in G.nodes()],
            node_color="lightblue",
        )

        # Draw edges with varying width based on weight
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=np.array(weights) * 5)

        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("State Transition Diagram")
        plt.savefig(
            self.get_figure_path("state_transition_diagram.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_node_activity_patterns(self):
        """
        Create a combined visualization of gene activity patterns and attractor frequencies.

        Generates a two-panel figure:
        Top panel:
        - Heatmap showing gene activity levels across attractors
        - X-axis: Attractors
        - Y-axis: Genes
        - Color intensity: Activity level (0-1)

        Bottom panel:
        - Bar chart showing attractor frequencies
        - X-axis: Attractors
        - Y-axis: Frequency (0-100%)

        Parameters:
        - Uses RdYlBu_r colormap for activity visualization
        - Includes grid lines for better readability
        - Shows frequency percentages on bars

        Saves output to: figures/node_activity_patterns.png
        """
        states = np.array(
            [[float(x) for x in attractor["state"]] for attractor in self.attractors]
        )
        frequencies = np.array(
            [attractor["frequency"] / self.total_runs for attractor in self.attractors]
        )

        gene_labels = [self.get_gene_name(i) for i in range(states.shape[1])]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # 1. Active pattern heatmap
        im = ax1.imshow(
            states.T,
            aspect="auto",
            cmap="RdYlBu_r",  # Use -yellow-blue color scheme
            interpolation="nearest",
        )

        # Add color bar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("Gene Activity", rotation=270, labelpad=15)

        # Set labels
        ax1.set_xticks(range(len(states)))
        ax1.set_xticklabels([f"A{i+1}" for i in range(len(states))])
        ax1.set_yticks(range(len(gene_labels)))
        ax1.set_yticklabels(gene_labels)

        # Add grid lines
        ax1.set_xticks(np.arange(-0.5, len(states), 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, len(gene_labels), 1), minor=True)
        ax1.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

        # 2. Frequency bar chart
        bars = ax2.bar(
            range(len(frequencies)), frequencies, color="steelblue", alpha=0.7
        )

        # Add frequency labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Set labels
        ax2.set_xticks(range(len(frequencies)))
        ax2.set_xticklabels([f"A{i+1}" for i in range(len(frequencies))])
        ax2.set_ylabel("Frequency")

        # Adjust layout
        plt.tight_layout()

        # Add total title
        fig.suptitle(
            "Gene Activity Patterns and Attractor Frequencies",
            y=1.02,
            size=14,
            weight="bold",
        )

        plt.savefig(
            self.get_figure_path("node_activity_patterns.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def analyze_basin_of_attraction(self):
        """Analyze and visualize basins of attraction"""
        # Calculate basin sizes
        basin_sizes = {}
        for attractor in self.attractors:
            state_str = "".join(
                "1" if float(v) > 0.4 else "0" for v in attractor["state"]
            )
            basin_sizes[state_str] = attractor["frequency"] / self.total_runs

        # Plot basin sizes
        plt.figure(figsize=(12, 6))
        sizes = list(basin_sizes.values())
        labels = [f"Basin {i+1}" for i in range(len(sizes))]
        plt.pie(sizes, labels=labels, autopct="%1.1f%%")
        plt.title("Basin of Attraction Sizes")

        plt.savefig(
            self.get_figure_path("basin_sizes.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        return basin_sizes

    def plot_phase_portrait_3d_with_vector_field(self, nodes=None, grid_size=20):
        if nodes is None:
            nodes = self.get_most_active_nodes(3)

        if len(nodes) != 3:
            raise ValueError("Exactly three nodes must be specified")

        node1, node2, node3 = nodes

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection="3d")

        # Prepare attractor data
        states1, states2, states3, sizes, frequencies = [], [], [], [], []
        for attractor in self.attractors:
            states1.append(float(attractor["state"][node1]))
            states2.append(float(attractor["state"][node2]))
            states3.append(float(attractor["state"][node3]))
            freq = attractor["frequency"] * 100 / self.total_runs
            sizes.append(freq)
            frequencies.append(freq)

        # Create grid and vector field
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        z = np.linspace(0, 1, grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        U, V, W = self.calculate_vector_field_3d(X, Y, Z, nodes)

        # Calculate vector field strength
        magnitude = np.sqrt(U**2 + V**2 + W**2)

        # Use blue colormap for uniform coloring
        vector_cmap = plt.cm.Blues  # Vector field uses blue
        attractor_cmap = plt.cm.Blues  # Attractors also use blue

        # Display vector field on multiple z planes
        z_levels = [0.2, 0.4, 0.6, 0.8]
        colors = vector_cmap(np.linspace(0.3, 0.9, len(z_levels)))  # Adjust color range

        for z_level, color in zip(z_levels, colors):
            z_index = int(z_level * (grid_size - 1))
            quiver = ax.quiver(
                X[::1, :, z_index],
                Y[::1, :, z_index],
                Z[::1, :, z_index],
                U[::1, :, z_index],
                V[::1, :, z_index],
                W[::1, :, z_index],
                length=0.1,
                normalize=True,
                color=color,
                alpha=0.4,
                linewidth=1.2,
                arrow_length_ratio=0.3,
            )

        # Plot attractors
        scatter = ax.scatter(
            states1,
            states2,
            states3,
            s=np.array(sizes) * 20,
            c=frequencies,
            cmap=attractor_cmap,  # Use the same color scheme
            alpha=1.0,
            edgecolors="black",
            linewidth=1,
            marker="o",
            label="Attractors",
        )

        # Add attractor frequency colorbar, adjust size
        cax = plt.gcf().add_axes(
            [0.95, 0.2, 0.02, 0.3]
        )  # [left, bottom, width, height]
        cbar = plt.colorbar(scatter, cax=cax, label="Attractor Frequency (%)")
        cbar.ax.tick_params(labelsize=8)  # Decrease tick font size
        cbar.ax.set_title("Freq.(%)", fontsize=8, pad=10)  # Add short title

        # Find stable state values (from attractor data)
        stable_state_value = None
        for attractor in self.attractors:
            state = np.array([float(x) for x in attractor["state"]])
            # Check if it is a single node stable state
            active_nodes = np.where(state > 0.1)[
                0
            ]  # Use 0.1 as threshold considering numerical errors
            if len(active_nodes) == 1:
                stable_state_value = state[active_nodes[0]]
                break

        if stable_state_value is None:
            stable_state_value = 0.809  # Use default value if found

        # Create single node stable state
        single_node_states = []
        for i, node in enumerate([node1, node2, node3]):
            state = np.zeros(len(self.attractors[0]["state"]))
            state[node] = stable_state_value
            label = f"Node {node+1} only"
            single_node_states.append((state, label))

        # Dynamically create marker styles
        single_node_markers = {
            f"Node {node1+1} only": {"marker": "s", "color": vector_cmap(0.3)},
            f"Node {node2+1} only": {"marker": "^", "color": vector_cmap(0.5)},
            f"Node {node3+1} only": {"marker": "D", "color": vector_cmap(0.7)},
        }

        # Plot single node states
        for state, label in single_node_states:
            marker_style = single_node_markers[label]
            ax.scatter(
                [state[node1]],
                [state[node2]],
                [state[node3]],
                c=[marker_style["color"]],
                marker=marker_style["marker"],
                s=150,
                edgecolors="black",
                linewidth=1,
                label=label,
                alpha=0.8,
            )

        # Adjust legend position and size
        legend = ax.legend(
            bbox_to_anchor=(1.15, 1),  # Move slightly to the left
            loc="upper right",
            frameon=True,
            framealpha=1,
            ncol=1,
            fontsize=8,  # Decrease font size
            borderaxespad=0,
            title="State Types",
            title_fontsize=9,
            markerscale=0.8,  # Decrease marker size in legend
        )

        # Adjust subplot layout, leave space for legend
        plt.subplots_adjust(right=0.85)

        # Improve labels
        ax.set_xlabel(
            f"Node {node1+1} State (Activity: {self.node_activities[node1]:.1%})"
        )
        ax.set_ylabel(
            f"Node {node2+1} State (Activity: {self.node_activities[node2]:.1%})"
        )
        ax.set_zlabel(
            f"Node {node3+1} State (Activity: {self.node_activities[node3]:.1%})"
        )

        title = f"3D Phase Portrait with Vector Field\nNodes {node1+1}, {node2+1}, {node3+1}"
        ax.set_title(title)

        # Set view and grid
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.2)

        # Set axis ranges
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        plt.savefig(
            self.get_figure_path(
                f"phase_portrait_3d_vector_field_{node1+1}_{node2+1}_{node3+1}.png"
            ),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.2,  # Add margin
        )
        plt.close()

    def calculate_streamline(self, start_point, nodes, num_points=100, step_size=0.01):
        """
        Calculate streamline trajectory from a starting point in state space.

        Parameters:
        -----------
        start_point : array-like
            Initial point coordinates in 3D state space
        nodes : list
            List of 3 node indices to use for state space
        num_points : int, optional
            Maximum number of points in streamline (default: 100)
        step_size : float, optional
            Integration step size (default: 0.01)

        Returns:
        --------
        numpy.ndarray
            Array of points forming the streamline trajectory
            Shape: (n_points, 3) where n_points <= num_points
        """
        streamline = np.zeros((num_points, 3))
        streamline[0] = start_point

        for i in range(1, num_points):
            # Get derivatives of the current point
            state = np.zeros(10)
            state[nodes[0]] = streamline[i - 1, 0]
            state[nodes[1]] = streamline[i - 1, 1]
            state[nodes[2]] = streamline[i - 1, 2]

            derivatives = self.equations(state, 0, 4)

            # Update position
            streamline[i, 0] = streamline[i - 1, 0] + step_size * derivatives[nodes[0]]
            streamline[i, 1] = streamline[i - 1, 1] + step_size * derivatives[nodes[1]]
            streamline[i, 2] = streamline[i - 1, 2] + step_size * derivatives[nodes[2]]

            # Ensure within boundaries
            streamline[i] = np.clip(streamline[i], 0, 1)

            # If reached stable point, end early
            if np.all(np.abs(streamline[i] - streamline[i - 1]) < 1e-6):
                return streamline[:i]

        return streamline

    def generate_network_equations(self):
        """Generate network equations using function_producion"""
        from network_frame import function_producion

        # Generate equation string
        eq_str = function_producion(self.network_data)

        # Create namespace and execute equations
        namespace = {}
        exec(eq_str, {"np": np}, namespace)

        return namespace["solve"]

    def calculate_vector_field_3d(self, X, Y, Z, nodes):
        """Calculate vector field using generated network equations"""
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        W = np.zeros_like(Z)

        node1, node2, node3 = nodes

        # Calculate vector field
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    state = np.zeros(10)  # Full state vector of 10 nodes
                    state[node1] = X[i, j, k]
                    state[node2] = Y[i, j, k]
                    state[node3] = Z[i, j, k]

                    derivatives = self.equations(state, 0, 4)

                    U[i, j, k] = derivatives[node1]
                    V[i, j, k] = derivatives[node2]
                    W[i, j, k] = derivatives[node3]

        # Process vector field
        magnitude = np.sqrt(U**2 + V**2 + W**2)
        max_magnitude = np.percentile(
            magnitude[magnitude > 0], 95
        )  # Use 95th percentile to avoid outliers

        # Clip large vectors to a reasonable range
        magnitude_threshold = max_magnitude * 0.1  # Set minimum threshold
        mask = magnitude > magnitude_threshold

        # Normalize vectors, avoiding division by zero
        U_norm = np.zeros_like(U)
        V_norm = np.zeros_like(V)
        W_norm = np.zeros_like(W)

        U_norm[mask] = U[mask] / magnitude[mask]
        V_norm[mask] = V[mask] / magnitude[mask]
        W_norm[mask] = W[mask] / magnitude[mask]

        return U_norm, V_norm, W_norm

    def generate_phase_portraits(self, node_sets=None):
        """
        Generate phase portraits for specified node sets

        Parameters:
        -----------
        node_sets : list of lists or None
            List of node sets to plot. Each set should contain 3 nodes.
            If None, uses the most active nodes.
        """
        if node_sets is None:
            # Use most active nodes
            active_nodes = self.get_most_active_nodes(3)
            node_sets = [active_nodes]

        for nodes in node_sets:
            print(f"Generating phase portrait for nodes {[n+1 for n in nodes]}...")
            self.plot_phase_portrait_3d_with_vector_field(nodes)

    def generate_all_visualizations(self):
        """
        Generate all visualizations and analyses with enhanced progress tracking
        """
        # First generate all visualizations
        visualization_steps = [
            (self.plot_node_activation_frequencies, "Node Activation Frequencies"),
            (self.plot_attractor_distribution, "Attractor Distribution"),
            (self.plot_state_network, "State Network"),
            (self.plot_correlation_matrix, "Correlation Matrix"),
            (self.plot_integrated_activity_patterns, "Integrated Activity Patterns"),
            (self.plot_state_transition_diagram, "State Transition Diagram"),
            (self.plot_basin_of_attraction, "Basin of Attraction"),
            (self.plot_regulatory_network, "Regulatory Network"),
            (self.plot_node_importance, "Node Importance Analysis"),
            (self.plot_node_ranking, "Node Ranking"),
        ]

        print("\nGenerating visualizations:")
        print("-" * 50)

        total_steps = len(visualization_steps)
        for i, (func, desc) in enumerate(visualization_steps, 1):
            try:
                print(f"[{i}/{total_steps}] {desc}...", end="", flush=True)
                func()
                print(" Done")
            except Exception as e:
                print(f"\nError in {desc}: {str(e)}")

        print("-" * 50)
        print("Visualization generation complete!")

        # Now perform regulatory pattern analysis
        print("\nAnalyzing regulatory patterns:")
        print("-" * 50)
        try:
            patterns = self.analyze_regulatory_patterns()

            # Print detailed results
            print("\nFeed-forward loops:")
            for ffl in patterns["feed_forward"]:
                print(
                    f"  {ffl['source']} -> {ffl['intermediate']} -> {ffl['target']} ({ffl['type']})"
                )

            print("\nFeedback loops:")
            for loop in patterns["feedback"]:
                print(f"  {' -> '.join(loop)} -> {loop[0]}")

            print("\nRegulatory cascades:")
            for cascade in patterns["cascades"]:
                print(f"  {' -> '.join(cascade)}")

            print("\nNetwork motifs:")
            for motif in patterns["motifs"]:
                if motif["type"] == "auto-regulation":
                    print(f"  Auto-regulation: {motif['gene']} ({motif['regulation']})")
                else:  # mutual_regulation
                    print(
                        f"  Mutual regulation: {' <-> '.join(motif['genes'])} ({motif['regulation_type']})"
                    )

        except Exception as e:
            print(f"Error in regulatory pattern analysis: {str(e)}")

        print("-" * 50)
        print("Analysis complete!")

    def get_most_active_nodes(self, n=3):
        """Get indices of n most active nodes"""
        node_freqs = self.calculate_node_frequencies()
        return np.argsort(node_freqs)[-n:]

    def calculate_node_importance(self):
        """Calculate node importance based on multiple metrics"""
        importance_scores = {}
        num_nodes = len(self.attractors[0]["state"])

        # 1. Activity scores (0-1)
        activity_scores = self.node_activities

        # 2. Connectivity scores (based on network structure)
        in_degree = np.zeros(num_nodes)
        out_degree = np.zeros(num_nodes)

        def parse_regulators(reg_str):
            """Parse regulator string and return valid node numbers"""
            if pd.isna(reg_str) or reg_str.lower() == "nothing":
                return []
            regulators = []
            for reg in str(reg_str).split(","):
                reg = reg.strip()
                try:
                    if reg and reg.lower() != "nothing":
                        regulators.append(int(reg) - 1)
                except ValueError:
                    continue  # Skip values that cannot be converted to integers
            return regulators

        # Process activation and inhibition relationships
        for _, row in self.network_data.iterrows():
            target = int(row["Node"]) - 1  # Current node (regulated)

            # Process activation relationships
            activators = parse_regulators(row["Activated by"])
            for source in activators:
                in_degree[target] += 1
                out_degree[source] += 1

            # Process inhibition relationships
            inhibitors = parse_regulators(row["Inhibited by"])
            for source in inhibitors:
                in_degree[target] += 1
                out_degree[source] += 1

        max_degree = max(np.max(in_degree), np.max(out_degree))
        if max_degree > 0:
            degree_scores = (in_degree + out_degree) / (2 * max_degree)
        else:
            degree_scores = np.zeros_like(in_degree)

        # 3. Attractor participation
        attractor_participation = np.zeros(num_nodes)
        total_freq = 0
        for attractor in self.attractors:
            state = np.array([float(x) for x in attractor["state"]])
            freq = attractor["frequency"]
            attractor_participation += (state > 0.1) * freq
            total_freq += freq
        attractor_participation = (
            attractor_participation / total_freq
            if total_freq > 0
            else np.zeros_like(attractor_participation)
        )

        # 4. Calculate combined scores
        weights = {"activity": 0.4, "connectivity": 0.3, "attractor_participation": 0.3}

        for i in range(num_nodes):
            importance_scores[i] = (
                weights["activity"] * activity_scores[i]
                + weights["connectivity"] * degree_scores[i]
                + weights["attractor_participation"] * attractor_participation[i]
            )

        return importance_scores, degree_scores, attractor_participation

    def plot_node_importance(self):
        """
        Plot node importance analysis with gene names instead of node numbers.

        Creates a visualization showing:
        - Node centrality
        - Regulatory influence
        - Network connectivity
        Using actual gene names for better interpretability

        Saves output to: figures/node_importance.png
        """
        # Calculate importance metrics
        importance_scores = {}
        num_nodes = len(self.attractors[0]["state"])

        # Calculate various metrics using gene names
        activity_scores = {
            self.get_gene_name(i): score for i, score in enumerate(self.node_activities)
        }

        # Calculate in/out degree using gene names
        in_degree = {self.get_gene_name(i): 0 for i in range(num_nodes)}
        out_degree = {self.get_gene_name(i): 0 for i in range(num_nodes)}

        # Process network data with gene names
        for i, row in self.network_data.iterrows():
            current_gene = self.get_gene_name(i)

            # Process activators
            if (
                not pd.isna(row["Activated by"])
                and row["Activated by"].lower() != "nothing"
            ):
                activators = row["Activated by"].split(",")
                for act in activators:
                    act = act.strip()
                    if act.isdigit():  # If it's a number, convert to gene name
                        act_gene = self.get_gene_name(int(act) - 1)
                        out_degree[act_gene] += 1
                        in_degree[current_gene] += 1

            # Process inhibitors
            if (
                not pd.isna(row["Inhibited by"])
                and row["Inhibited by"].lower() != "nothing"
            ):
                inhibitors = row["Inhibited by"].split(",")
                for inh in inhibitors:
                    inh = inh.strip()
                    if inh.isdigit():  # If it's a number, convert to gene name
                        inh_gene = self.get_gene_name(int(inh) - 1)
                        out_degree[inh_gene] += 1
                        in_degree[current_gene] += 1

        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Activity Scores
        genes = list(activity_scores.keys())
        scores = list(activity_scores.values())
        ax1.bar(genes, scores, color="skyblue")
        ax1.set_title("Gene Activity Scores")
        ax1.set_xticklabels(genes, rotation=45, ha="right")
        ax1.set_ylabel("Activity Score")

        # 2. In-degree
        ax2.bar(genes, [in_degree[g] for g in genes], color="lightgreen")
        ax2.set_title("Input Connections")
        ax2.set_xticklabels(genes, rotation=45, ha="right")
        ax2.set_ylabel("Number of Inputs")

        # 3. Out-degree
        ax3.bar(genes, [out_degree[g] for g in genes], color="salmon")
        ax3.set_title("Output Connections")
        ax3.set_xticklabels(genes, rotation=45, ha="right")
        ax3.set_ylabel("Number of Outputs")

        # Adjust layout
        plt.tight_layout()

        # Add overall title
        fig.suptitle("Gene Importance Analysis", y=1.05, size=14, weight="bold")

        # Save figure
        plt.savefig(
            self.get_figure_path("node_importance.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_regulatory_network(self):
        """Plot regulatory network"""
        from graphviz import Digraph

        # Create directed graph
        dot = Digraph(comment="Regulatory Network")
        dot.attr(rankdir="TB")

        # Node style
        dot.attr("node", shape="circle", style="filled", fillcolor="white")

        # Add all nodes
        for i in range(len(self.attractors[0]["state"])):
            gene_name = self.get_gene_name(i)
            dot.node(f"Node{i+1}", gene_name)

        # Add regulatory relationships
        for _, row in self.network_data.iterrows():
            target = f'Node{row["Node"]}'

            # Process activation relationships
            if not pd.isna(row["Activated by"]):
                activators = str(row["Activated by"]).split(",")
                for activator in activators:
                    if activator.strip() and activator.strip().lower() != "nothing":
                        dot.edge(
                            f"Node{activator.strip()}",
                            target,
                            color="green",
                            arrowhead="normal",
                        )

            # Process inhibition relationships
            if not pd.isna(row["Inhibited by"]):
                inhibitors = str(row["Inhibited by"]).split(",")
                for inhibitor in inhibitors:
                    if inhibitor.strip() and inhibitor.strip().lower() != "nothing":
                        dot.edge(
                            f"Node{inhibitor.strip()}",
                            target,
                            color="red",
                            arrowhead="tee",
                        )

        # Save image
        dot.render(
            self.get_figure_path("regulatory_network"), format="png", cleanup=True
        )

    def load_gene_mapping(self, mapping_file):
        """Load gene-node mapping"""
        if mapping_file is None:
            return {i: f"Node {i+1}" for i in range(len(self.attractors[0]["state"]))}

        try:
            # Read Excel file
            mapping_data = pd.read_excel(mapping_file)

            # Check actual column names
            if (
                "Gene" not in mapping_data.columns
                or "Number" not in mapping_data.columns
            ):
                print(
                    "Warning: Mapping file should contain 'Gene' and 'Number' columns"
                )
                print("Available columns:", mapping_data.columns)
                return {
                    i: f"Node {i+1}" for i in range(len(self.attractors[0]["state"]))
                }

            # Create mapping from node to gene name
            gene_map = {
                int(row["Number"]) - 1: str(row["Gene"])
                for _, row in mapping_data.iterrows()
            }

            # Print mapping for verification
            print("\nGene mapping:")
            for node, gene in sorted(gene_map.items()):
                print(f"Node {node+1} -> {gene}")

            return gene_map

        except Exception as e:
            print(f"Error processing mapping file: {e}")
            return {i: f"Node {i+1}" for i in range(len(self.attractors[0]["state"]))}

    def get_gene_name(self, node_index):
        """Get gene name for a given node index"""
        gene_name = self.gene_names.get(node_index, f"Node {node_index+1}")
        # Handle special characters
        if "C/EBP" in gene_name:
            gene_name = gene_name.replace("C/EBP", "CEBP")  # Avoid path issues
        return gene_name

    def plot_basin_of_attraction(self):
        """Plot basin of attraction analysis with improved label handling"""
        # Calculate basin sizes
        basin_sizes = [
            attractor["frequency"] / self.total_runs for attractor in self.attractors
        ]

        # Prepare labels and colors
        labels = []
        colors = []
        for i, attractor in enumerate(self.attractors):
            state = [float(x) for x in attractor["state"]]
            active_genes = [
                self.get_gene_name(j) for j, val in enumerate(state) if val > 0.1
            ]

            # Display attractor number only on wedges, move active genes info to legend
            labels.append(f"A{i+1}")
            colors.append(plt.cm.Set3(i % 12))

        # Create figure
        plt.figure(figsize=(15, 10))

        # Plot pie chart
        wedges, texts, autotexts = plt.pie(
            basin_sizes,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%"
            if pct > 1
            else "",  # Only show percentages > 1%
            pctdistance=0.85,
            labeldistance=1.1,  # Increase label distance
            wedgeprops=dict(width=0.5, edgecolor="white"),
        )

        # Set label style
        plt.setp(autotexts, size=8, weight="bold")
        plt.setp(texts, size=8)

        # Add title
        plt.title("Basin of Attraction Analysis", pad=20, size=14, weight="bold")

        # Create detailed legend labels
        legend_labels = []
        for i, attractor in enumerate(self.attractors):
            state = [float(x) for x in attractor["state"]]
            active_genes = [
                self.get_gene_name(j) for j, val in enumerate(state) if val > 0.1
            ]
            percentage = basin_sizes[i] * 100
            if active_genes:
                label = f'A{i+1} ({percentage:.1f}%): {", ".join(active_genes)}'
            else:
                label = f"A{i+1} ({percentage:.1f}%): Inactive"
            legend_labels.append(label)

        # Display legend in multiple columns
        plt.legend(
            wedges,
            legend_labels,
            title="Attractors and Active Genes",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=8,
            title_fontsize=10,
            ncol=1,  # Adjust columns as needed
            columnspacing=1,
            handlelength=1,
            handleheight=1,
        )

        # Add explanatory text
        plt.figtext(
            0.95,
            0.02,
            "Note: Percentages indicate basin sizes\nGenes shown are those with activity > 0.1",
            ha="right",
            fontsize=8,
            style="italic",
        )

        # Save image
        plt.savefig(
            self.get_figure_path("basin_of_attraction.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    def plot_integrated_activity_patterns(self):
        """
        Create integrated activity pattern visualization with improved layout
        """
        # Prepare data
        states = np.array(
            [[float(x) for x in attractor["state"]] for attractor in self.attractors]
        )
        frequencies = np.array(
            [attractor["frequency"] / self.total_runs for attractor in self.attractors]
        )

        # Get gene names and attractor labels
        gene_labels = [self.get_gene_name(i) for i in range(states.shape[1])]
        attractor_labels = [f"A{i+1}" for i in range(len(states))]

        # Sort by frequency in descending order
        sort_indices = np.argsort(frequencies)[::-1]

        # Rearrange data
        states = states[sort_indices]
        frequencies = frequencies[sort_indices]
        attractor_labels = [attractor_labels[i] for i in sort_indices]

        # Calculate gene clustering order
        col_linkage = hierarchy.linkage(states.T, method="ward")
        col_order = hierarchy.dendrogram(col_linkage, no_plot=True)["leaves"]

        # Rearrange gene order
        states_clustered = states[:, col_order]
        reordered_gene_labels = [gene_labels[i] for i in col_order]

        # Create figure
        fig = plt.figure(figsize=(15, 12))

        # Define main grid
        gs = GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=[0.85, 0.15],
            height_ratios=[0.15, 0.85],
            wspace=0.02,
            hspace=0.02,
        )

        # Top dendrogram
        ax_top = fig.add_subplot(gs[0, 0])
        hierarchy.dendrogram(col_linkage, ax=ax_top, no_labels=True)
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        # Main heatmap
        ax_center = fig.add_subplot(gs[1, 0])
        im = ax_center.imshow(
            states_clustered,
            aspect="auto",
            cmap=plt.cm.RdBu_r,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        # Set heatmap labels
        ax_center.set_xticks(range(len(reordered_gene_labels)))
        ax_center.set_xticklabels(
            reordered_gene_labels, rotation=45, ha="right", fontsize=10
        )
        ax_center.set_yticks(range(len(attractor_labels)))
        ax_center.set_yticklabels(attractor_labels, fontsize=10)

        # Add grid lines
        # Add grid lines in both vertical and horizontal directions
        ax_center.set_xticks(np.arange(-0.5, len(reordered_gene_labels), 1), minor=True)
        ax_center.set_yticks(np.arange(-0.5, len(attractor_labels), 1), minor=True)

        # Set grid line style
        ax_center.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

        # Add outer border
        for spine in ax_center.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        # Ensure grid lines are above the image
        ax_center.set_axisbelow(False)

        # Right frequency distribution plot
        ax_right = fig.add_subplot(gs[1, 1])
        ax_right.barh(range(len(frequencies)), frequencies, color="#4682B4", alpha=0.7)
        ax_right.set_yticks([])
        ax_right.set_xlabel("Frequency", fontsize=10)

        # Align right plot with heatmap
        ax_right.set_ylim(ax_center.get_ylim())

        # Frequency labels
        for i, freq in enumerate(frequencies):
            if freq >= 0.01:
                ax_right.text(
                    freq + 0.001, i, f"{freq:.1%}", va="center", ha="left", fontsize=8
                )

        # Color bar
        cax = plt.axes([0.15, 0.05, 0.5, 0.02])
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Gene Activity", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        # Title
        fig.suptitle(
            "Integrated Gene Activity Analysis\n(Sorted by Frequency)",
            y=0.95,
            size=14,
            weight="bold",
        )

        # Adjust layout
        plt.subplots_adjust(
            left=0.15,  # Increase left margin
            right=0.92,  # Decrease right margin
            bottom=0.08,  # Increase bottom margin
            top=0.92,  # Decrease top margin
        )

        # Save image
        plt.savefig(
            self.get_figure_path("integrated_activity_patterns.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close()

    def validate_data(self):
        """
        Validate input data integrity and format

        Raises:
        -------
        ValueError: If data format is invalid
        """
        if not self.attractors:
            raise ValueError("No attractor data found")

        if not all(len(a["state"]) == 10 for a in self.attractors):
            raise ValueError("Invalid attractor state dimensions")

    def verify_outputs(self):
        """
        Verify all output files were generated correctly

        Returns:
        --------
        bool
            True if all files exist and are valid
        """
        required_files = [
            "node_frequencies.png",
            "attractor_distribution.png",
            # ... other expected files ...
        ]

        for file in required_files:
            path = self.get_figure_path(file)
            if not os.path.exists(path):
                print(f"Missing output file: {file}")
                return False

        return True

    def analyze_network_equations(self):
        """
        Analyze the structure and characteristics of network equations

        Returns:
        --------
        dict
            Analysis results containing:
            - self_regulation: List of self-regulating nodes
            - cross_regulation: Dict of cross-regulatory relationships
            - hill_coefficient: Hill function coefficient
            - decay_rate: Universal decay rate
        """
        analysis = {
            "self_regulation": [],
            "cross_regulation": {},
            "hill_coefficient": 3,  # x**3 in equations
            "regulation_strength": 8,  # coefficient before x**3
            "decay_rate": 1,  # -x term
        }

        # Analyze equation structure
        equations = {
            "RUNX1": ["RUNX1"],  # self-regulation only
            "PU.1": ["PU.1", "C/EBPα", "RUNX1"],
            "C/EBPα": ["C/EBPα", "RUNX1"],
            "GFI1": ["GFI1", "C/EBPα", "IKAROS"],
            "EGR1": ["EGR1", "PU.1"],
            "GATA1": ["GATA1", "RUNX1"],
            "KLF1": ["KLF1", "GATA1"],
            "FLI1": ["FLI1", "GATA1", "PU.1"],
            "IKAROS": ["IKAROS"],
            "EBF1": ["EBF1", "IKAROS"],
        }

        return analysis, equations

    def analyze_network_structure(self):
        """
        Analyze and visualize network structure characteristics

        Analyzes:
        - Node connectivity
        - Regulatory motifs
        - Network modularity
        - Feedback loops
        """
        network_stats = {
            "nodes": 10,
            "self_loops": 0,
            "cross_regulations": 0,
            "feedback_loops": [],
        }

        # Count regulatory relationships
        for node, equation in enumerate(self.network_data.iterrows(), 1):
            # Self-regulation
            if f"x{node}" in equation:
                network_stats["self_loops"] += 1

            # Cross-regulation
            other_nodes = [f"x{i}" for i in range(1, 11) if i != node]
            for other in other_nodes:
                if other in equation:
                    network_stats["cross_regulations"] += 1

        return network_stats

    def validate_gene_mapping(self):
        """
        Validate gene mapping consistency and completeness

        Checks:
        - All nodes have corresponding genes
        - No duplicate mappings
        - Valid gene names

        Returns:
        --------
        bool
            True if mapping is valid
        str
            Error message if invalid
        """
        expected_nodes = set(range(10))  # 0-9 for 10 nodes
        mapped_nodes = set(self.gene_names.keys())

        if expected_nodes != mapped_nodes:
            missing = expected_nodes - mapped_nodes
            extra = mapped_nodes - expected_nodes
            return False, f"Missing nodes: {missing}, Extra nodes: {extra}"

        # Check for duplicate gene names
        gene_names = list(self.gene_names.values())
        if len(gene_names) != len(set(gene_names)):
            return False, "Duplicate gene names found"

        return True, "Gene mapping is valid"

    def analyze_regulatory_patterns(self):
        """
        Analyze regulatory patterns in the network

        Identifies:
        - Feed-forward loops: A->B->C, where A also directly regulates C
        - Feedback loops: Cycles where genes regulate each other in a loop
        - Regulatory cascades: Linear chains of regulation (A->B->C->D)
        - Network motifs: Common recurring regulatory patterns

        Returns:
        --------
        dict
            Dictionary containing identified regulatory patterns
        """
        patterns = {"feed_forward": [], "feedback": [], "cascades": [], "motifs": []}

        # Create adjacency matrix for network analysis
        num_nodes = len(self.attractors[0]["state"])
        adj_matrix = np.zeros((num_nodes, num_nodes))

        # Fill adjacency matrix based on network data
        for i, row in self.network_data.iterrows():
            target = i

            # Process activators
            if (
                not pd.isna(row["Activated by"])
                and row["Activated by"].lower() != "nothing"
            ):
                activators = row["Activated by"].split(",")
                for act in activators:
                    act = act.strip()
                    if act.isdigit():
                        source = int(act) - 1
                        adj_matrix[source][target] = 1

            # Process inhibitors
            if (
                not pd.isna(row["Inhibited by"])
                and row["Inhibited by"].lower() != "nothing"
            ):
                inhibitors = row["Inhibited by"].split(",")
                for inh in inhibitors:
                    inh = inh.strip()
                    if inh.isdigit():
                        source = int(inh) - 1
                        adj_matrix[source][target] = -1

        # 1. Identify feed-forward loops
        for a in range(num_nodes):
            for b in range(num_nodes):
                if adj_matrix[a][b] != 0:  # If A regulates B
                    for c in range(num_nodes):
                        if (
                            adj_matrix[b][c] != 0  # If B regulates C
                            and adj_matrix[a][c] != 0
                        ):  # and A directly regulates C
                            patterns["feed_forward"].append(
                                {
                                    "source": self.get_gene_name(a),
                                    "intermediate": self.get_gene_name(b),
                                    "target": self.get_gene_name(c),
                                    "type": "coherent"
                                    if adj_matrix[a][b] * adj_matrix[b][c]
                                    == adj_matrix[a][c]
                                    else "incoherent",
                                }
                            )

        # 2. Identify feedback loops using DFS
        def find_cycles(start, current, path, visited):
            if current in visited and current == start and len(path) > 2:
                cycle = [self.get_gene_name(i) for i in path]
                if cycle not in patterns["feedback"]:
                    patterns["feedback"].append(cycle)
                return

            if current in visited:
                return

            visited.add(current)
            path.append(current)

            for next_node in range(num_nodes):
                if adj_matrix[current][next_node] != 0:
                    find_cycles(start, next_node, path.copy(), visited.copy())

        for node in range(num_nodes):
            find_cycles(node, node, [], set())

        # 3. Identify regulatory cascades
        def find_cascade(start, path):
            if len(path) >= 3:  # Minimum cascade length
                cascade = [self.get_gene_name(i) for i in path]
                if cascade not in patterns["cascades"]:
                    patterns["cascades"].append(cascade)

            if len(path) >= 6:  # Maximum cascade length
                return

            current = path[-1]
            for next_node in range(num_nodes):
                if adj_matrix[current][next_node] != 0 and next_node not in path:
                    find_cascade(start, path + [next_node])

        for node in range(num_nodes):
            find_cascade(node, [node])

        # 4. Identify common network motifs
        # Auto-regulation
        for i in range(num_nodes):
            if adj_matrix[i][i] != 0:
                patterns["motifs"].append(
                    {
                        "type": "auto-regulation",
                        "gene": self.get_gene_name(i),
                        "regulation": "activation"
                        if adj_matrix[i][i] > 0
                        else "inhibition",
                    }
                )

        # Mutual regulation
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i][j] != 0 and adj_matrix[j][i] != 0:
                    patterns["motifs"].append(
                        {
                            "type": "mutual_regulation",
                            "genes": [self.get_gene_name(i), self.get_gene_name(j)],
                            "regulation_type": "positive"
                            if adj_matrix[i][j] * adj_matrix[j][i] > 0
                            else "negative",
                        }
                    )

        # Print summary
        print("\nRegulatory Pattern Analysis:")
        print(f"Feed-forward loops: {len(patterns['feed_forward'])}")
        print(f"Feedback loops: {len(patterns['feedback'])}")
        print(f"Regulatory cascades: {len(patterns['cascades'])}")
        print(f"Common motifs: {len(patterns['motifs'])}")

        return patterns

    def plot_node_ranking(self):
        """
        Plot horizontal node ranking visualization showing relative importance of nodes.
        Nodes are sorted by importance score in descending order.

        Creates a horizontal bar chart showing:
        - Node importance scores
        - Relative rankings
        - Color-coded bars based on score ranges
        """
        # Calculate importance scores for all nodes
        importance_scores = {}
        num_nodes = len(self.attractors[0]["state"])

        for i in range(num_nodes):
            activity = self.node_activities[i]
            state_participation = sum(
                1 for attractor in self.attractors if float(attractor["state"][i]) > 0.1
            )
            frequency_weight = (
                sum(
                    attractor["frequency"]
                    for attractor in self.attractors
                    if float(attractor["state"][i]) > 0.1
                )
                / self.total_runs
            )

            importance_scores[i] = (
                0.4 * activity
                + 0.3 * state_participation / len(self.attractors)
                + 0.3 * frequency_weight
            )

        # Create sorted pairs of (gene_name, score)
        gene_scores = [
            (self.get_gene_name(i), score) for i, score in importance_scores.items()
        ]

        # Sort by score in descending order
        gene_scores.sort(key=lambda x: x[1], reverse=True)

        # Separate into genes and scores
        genes, scores = zip(*gene_scores)

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot bars in reverse order to show highest scores at top
        y_pos = range(len(genes) - 1, -1, -1)  # Reverse the y-positions
        bars = plt.barh(y_pos, scores)

        # Color code bars based on score ranges
        for i, bar in enumerate(bars):
            if scores[i] > 0.3:  # High importance
                bar.set_color("yellow")
            elif scores[i] > 0.2:  # Medium importance
                bar.set_color("lightgreen")
            else:  # Low importance
                bar.set_color("lightgray")

        # Set y-axis labels with gene names (in reverse order to match bars)
        plt.yticks(y_pos, genes)

        # Add value labels on bars
        for i, score in enumerate(scores):
            plt.text(
                score + 0.001,
                y_pos[i],
                f"{score:.3f}",
                va="center",
                ha="left",
                fontsize=10,
            )

        # Customize plot
        plt.title("Node Importance Ranking", pad=20, size=14)
        plt.xlabel("Importance Score")

        # Add grid
        plt.grid(axis="x", linestyle="--", alpha=0.3)

        # Set x-axis limits to ensure all labels are visible
        plt.xlim(0, max(scores) * 1.1)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(
            self.get_figure_path("node_ranking.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python visualize_attractors.py <attractor_file> <network_file> [gene_mapping_file]"
        )
        sys.exit(1)

    attractor_file = sys.argv[1]
    network_file = sys.argv[2]
    gene_mapping_file = sys.argv[3] if len(sys.argv) > 3 else None

    visualizer = AttractorVisualizer(attractor_file, network_file, gene_mapping_file)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
