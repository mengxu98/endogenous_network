# -*- coding: utf-8 -*-
"""
@title Endogenous Network Simulation with Attractor Calculation
@auther mengxu
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import os
from multiprocessing import Pool
import random
import time
import argparse


def solve(w, t, n, frame):
    dxdt = []
    namespace = {f"x{i+1}": w[i] for i in range(len(w))}

    for idx, row in frame.iterrows():
        activated_by = str(row["Activated by"])
        inhibited_by = str(row["Inhibited by"])
        node_id = row["Node"]

        if activated_by != "nothing":
            if "," in activated_by:
                activators = activated_by.split(",")
                activation = "+".join([f"x{a.strip()}**3" for a in activators])
            else:
                activation = f"x{activated_by.strip()}**3"
            activation = f"8*({activation}) / (1 + 8*({activation}))"
        else:
            activation = "1"

        if inhibited_by != "nothing":
            if "," in inhibited_by:
                inhibitors = inhibited_by.split(",")
                inhibition = "*".join(
                    [f"1 / (1 + 8*(x{i.strip()}**3))" for i in inhibitors]
                )
            else:
                inhibition = f"1 / (1 + 8*(x{inhibited_by.strip()}**3))"
        else:
            inhibition = "1"

        expr = f"{activation} * {inhibition} - x{node_id}"
        dxdt.append(eval(expr, {"__builtins__": None}, namespace))

    return np.array(dxdt)


def solve_dynamics(w, t, n, frame):
    """
    Solve the differential equations for the given state and frame.
    """
    return solve(w, t, n, frame)


def calculate_attractor(args):
    """
    Modified to accept a single argument tuple for multiprocessing compatibility
    """
    frame, start_list = args
    t = np.arange(0, 100, 1)
    if start_list is None:
        num_nodes = len(frame)
        start = np.random.random(num_nodes)
    else:
        start = start_list

    track = odeint(lambda w, t, n: solve_dynamics(w, t, n, frame), start, t, args=(4,))
    final_state = track[-1]
    return ["%.4f" % abs(x) for x in final_state]


def analyze_attractors(frame, outputfile, start_list, num_iterations=2000):
    """
    Run multiple times to analyze attractor distribution.
    """
    attractors = []
    attractor_dict = {}

    with Pool() as pool:
        args = [(frame, start_list) for _ in range(num_iterations)]
        results = pool.map(calculate_attractor, args)
        pool.close()
        pool.join()

        for attractor in results:
            attractor = tuple(attractor)
            if attractor not in attractor_dict:
                attractor_dict[attractor] = 1
                attractors.append(attractor)
            else:
                attractor_dict[attractor] += 1

    output_path = os.path.join(os.getcwd(), outputfile)
    with open(output_path, "w") as f:
        for attractor, count in attractor_dict.items():
            f.write(f"{count}\t{attractor}\n")

    print(f"Number of unique attractors: {len(attractors)}")
    return attractors


def read_network_from_txt(filename):
    """
    Read network structure from txt file.
    Expected format:
    Node    Activated by    Inhibited by
    1       2,3            4
    2       1              nothing
    ...
    """
    data = []
    with open(filename, "r") as f:
        # Skip header
        next(f)
        for line in f:
            node, activated, inhibited = line.strip().split("\t")
            data.append(
                {
                    "Node": int(node),
                    "Activated by": activated,
                    "Inhibited by": inhibited,
                }
            )
    return pd.DataFrame(data)


def main(
    inputfile="network_without_0.txt", outputfile="attractors.txt", start_list=None
):
    """
    Read the input file, generate the dynamics model, and analyze attractors.

    Args:
        inputfile (str): Path to the input network file
        outputfile (str): Path to save the attractors results (without extension)
        start_list (list, optional): Initial state list. Defaults to None.
    """
    frame = read_network_from_txt(inputfile)
    attractors = analyze_attractors(frame, outputfile, start_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endogenous Network Simulation")
    parser.add_argument(
        "-i",
        "--input",
        default="network_without_0.txt",
        help="Input network file path (default: network_without_0.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="attractors.txt",
        help="Output file path without extension (default: attractors.txt)",
    )

    args = parser.parse_args()
    main(args.input, args.output)
