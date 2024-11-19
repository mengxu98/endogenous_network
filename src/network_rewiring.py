# -*- coding: utf-8 -*-
"""
@title Network Rewiring
@author mengxu
"""

import pandas as pd
import sys, os
import argparse

# Define project root at the beginning of the file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Get parent directory

# Read the original network data
input_file = "network.txt"
file_path = os.path.join(current_dir, "data", input_file)


def create_gene_mapping(frame):
    """Create a mapping between gene names and numbers"""
    # Get gene names from the Node column
    genes = frame["Node"].tolist()
    # Clean spaces from gene names
    genes = [str(gene).strip() for gene in genes]

    # Create mappings
    gene_to_num = {gene: str(i) for i, gene in enumerate(genes, 1)}
    num_to_gene = {str(i): gene for i, gene in enumerate(genes, 1)}

    # Save mapping to file
    mapping_df = pd.DataFrame(list(gene_to_num.items()), columns=["Gene", "Number"])
    mapping_file = os.path.join("results", "gene_mapping.txt")
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    mapping_df.to_csv(mapping_file, sep="\t", index=False)

    print("\nGene to number mapping:")
    print(mapping_df)

    return gene_to_num, num_to_gene


def clean_gene_name(name):
    """Clean gene names by removing extra spaces and standardizing format"""
    if pd.isna(name):
        return name
    # Remove leading/trailing spaces
    name = str(name).strip()
    # Remove extra spaces within the name
    name = " ".join(name.split())
    # Remove any spaces between letters/numbers
    parts = name.split(" ")
    if len(parts) > 1:
        # If the last part is a number, join it with the previous part
        if parts[-1].isdigit():
            name = "".join(parts)
    return name


def convert_frame_to_numbers(frame, gene_to_num):
    # Create a copy of the frame
    frame_num = frame.copy()
    frame_num.set_index("Node", inplace=True)

    # Convert activators
    def convert_activators(act_str):
        if pd.isna(act_str):
            return "nothing"
        acts = str(act_str).split(",")
        # Clean each gene name
        acts = [clean_gene_name(gene.strip()) for gene in acts]
        try:
            return ",".join(str(gene_to_num[gene]) for gene in acts)
        except KeyError as e:
            print(f"Warning: Gene not found in mapping: {e}")
            return "nothing"

    # Convert inhibitors
    def convert_inhibitors(inh_str):
        if pd.isna(inh_str):
            return "nothing"
        inhs = str(inh_str).split(",")
        # Clean each gene name
        inhs = [clean_gene_name(gene.strip()) for gene in inhs]
        try:
            return ",".join(str(gene_to_num[gene]) for gene in inhs)
        except KeyError as e:
            print(f"Warning: Gene not found in mapping: {e}")
            return "nothing"

    # Apply conversions
    frame_num["Activated by"] = frame_num["Activated by"].apply(convert_activators)
    frame_num["Inhibited by"] = frame_num["Inhibited by"].apply(convert_inhibitors)

    # Convert index (Node names) to numbers
    frame_num.index = frame_num.index.map(lambda x: gene_to_num[clean_gene_name(x)])

    return frame_num


def network_rewiring(num, frame=None):
    """Remove a node and its connections from the network"""
    if frame is None:
        frame = globals()["frame"]
    frame = frame.fillna("nothing")
    frame = frame.drop(num)
    for i in frame.index:
        for line in ["Activated by", "Inhibited by"]:
            if pd.isna(frame.loc[i, line]) or frame.loc[i, line] == "":
                frame.loc[i, line] = "nothing"
            if str(frame.loc[i, line]) != "nothing":
                if "," in str(frame.loc[i, line]):
                    lst = str(frame.loc[i, line]).split(",")
                    if str(num) in lst:
                        lst.remove(str(num))
                    if len(lst) != 0:
                        frame.loc[i, line] = ",".join(lst)
                    else:
                        frame.loc[i, line] = "nothing"
                else:
                    if str(frame.loc[i, line]) == str(num):
                        frame.loc[i, line] = "nothing"
    return frame


def build(args):
    """Build new network by removing specified nodes"""
    i, nodesnum = args
    frame_temp = network_rewiring(i[0])
    l = len(i)
    for time in range(1, l):
        try:
            frame_temp = network_rewiring(i[time], frame_temp)
        except:
            pass
        # Remove isolated nodes
        for ind in frame_temp.index:
            if (frame_temp.loc[ind]["Activated by"] == "nothing") and (
                frame_temp.loc[ind]["Inhibited by"] == "nothing"
            ):
                try:
                    frame_temp = network_rewiring(ind, frame_temp)
                except:
                    pass

    # Last ensure no empty values
    frame_temp = frame_temp.fillna("nothing")
    frame_temp = frame_temp.replace("", "nothing")

    # Create output DataFrame with node numbers
    output_frame = pd.DataFrame()
    output_frame["Node"] = frame_temp.index
    output_frame["Activated by"] = frame_temp["Activated by"]
    output_frame["Inhibited by"] = frame_temp["Inhibited by"]

    # Generate filename from the nodes being removed
    filename = [str(x) for x in i]  # Fix: generate filename

    # Save to results folder with headers
    results_path = os.path.join(project_root, "results", str(nodesnum))
    os.makedirs(results_path, exist_ok=True)

    output_file = os.path.join(
        results_path, "network_without_" + "_".join(filename) + ".txt"
    )
    output_frame.to_csv(output_file, sep="\t", index=False)


def save_network(network_data, output_dir, removed_nodes=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine filename based on removed nodes
    if not removed_nodes:
        filename = "network_without_0.txt"
    else:
        nodes_str = "_".join(map(str, removed_nodes))
        filename = f"network_without_{nodes_str}.txt"

    # Save to the specified directory
    output_path = os.path.join(output_dir, filename)
    network_data.to_csv(output_path, sep="\t", index=False)
    print(f"Network saved to: {output_path}")


def main(nodesnum, totalnum):
    import itertools as it

    # Generate all possible combinations of nodes to remove
    node_range = frame.index.tolist()  # Get actual node list
    combinations = list(it.combinations(node_range, nodesnum))

    # Create results directory if it doesn't exist
    results_path = os.path.join(project_root, "results", str(nodesnum))
    os.makedirs(results_path, exist_ok=True)

    # Prepare arguments for build function
    args = [(comb, nodesnum) for comb in combinations]

    import time

    time1 = time.time()
    from multiprocessing import Pool as ThreadPool

    with ThreadPool(processes=4) as pool:  # Limit number of processes to 4
        pool.map(build, args)
    time2 = time.time()
    print(f"Processing time: {time2 - time1:.2f} seconds")

    # Convert data frame to number version
    frame_num = convert_frame_to_numbers(frame, gene_to_num)
    print("\nConverted data (with numbers):")
    print(frame_num.head())

    # Use number version of data frame for further processing
    frame = frame_num

    if len(sys.argv) < 2:
        print("Please provide a number as argument:")
        print("0: Run original network")
        print("n: Remove n nodes from network")
        print("Example: python network_rewiring.py 2")
        sys.exit(1)

    if sys.argv[1] == str(0):
        print("\nOriginal network is running")
        frame = frame.fillna("nothing")

        # Print output data
        print("\nFinal data to be saved:")
        print(frame.head())

        # Create specified directory structure
        results_path = os.path.join(project_root, "results")
        results_dir = os.path.join(results_path, "0")

        print(f"\nCreating directories:")
        print(f"Results path: {results_path}")
        print(f"Results dir: {results_dir}")

        # Create directories
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Save to specified directory
        output_file = os.path.join(results_dir, "network_without_0.txt")
        try:
            # Save number version of frame
            output_df = frame.reset_index()
            output_df.to_csv(output_file, sep="\t", index=False)
            print(f"\nSaved network to: {output_file}")

            # Verify file
            if os.path.exists(output_file):
                print(f"File exists and size is: {os.path.getsize(output_file)} bytes")
                # Read and print saved file contents
                saved_df = pd.read_csv(output_file, sep="\t")
                print("\nSaved file contents:")
                print(saved_df.head())
            else:
                print("Warning: File was not created!")

            # Save mapping file to results directory
            mapping_file = os.path.join(results_path, "gene_mapping.txt")
            mapping_df.to_csv(mapping_file, sep="\t", index=False)
            print(f"\nSaved mapping to: {mapping_file}")

        except Exception as e:
            print(f"Error saving file: {str(e)}")
            print("Full path details:")
            print(f"Current directory: {os.getcwd()}")
            print(f"Project root: {project_root}")
            print(f"Results path exists: {os.path.exists(results_path)}")
            print(f"Results dir exists: {os.path.exists(results_dir)}")

        # Save the network data to the specified directory
        save_network(frame, f"results/{args.nodes}")


def parse_args():
    parser = argparse.ArgumentParser(description="Network rewiring analysis")
    parser.add_argument("input_dir", help="Input directory containing network.txt")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument(
        "nodes", type=int, help="Number of nodes to remove (0 for original network)"
    )
    return parser.parse_args()


def read_network_file(file_path):
    """Try different encodings to read the network file"""
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]

    for encoding in encodings:
        try:
            print(f"Trying to read with {encoding} encoding...")
            # First try reading with tab separator
            df = pd.read_csv(file_path, sep="\t", encoding=encoding)
            if len(df.columns) < 2:  # If we got only one column, try comma separator
                df = pd.read_csv(file_path, sep=",", encoding=encoding)
            return df
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding")
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {str(e)}")
            continue

    raise ValueError(
        f"Could not read file {file_path} with any of the attempted encodings"
    )


if __name__ == "__main__":
    import traceback

    try:
        # Parse command line arguments
        args = parse_args()

        # Setup input/output paths
        file_path = os.path.join(args.input_dir, input_file)
        project_root = args.output_dir

        # Verify file exists
        if not os.path.exists(file_path):
            print(f"Error: Input file not found at {file_path}")
            sys.exit(1)

        # Print file info
        print(f"File path: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")

        # Read the original network data
        print(f"Trying to read file from: {file_path}")
        frame = read_network_file(file_path)

        # Verify data was read correctly
        print("\nDataFrame info:")
        print(frame.info())
        print("\nFirst few rows:")
        print(frame.head())

        # Clean gene names in the Node column
        frame["Node"] = frame["Node"].apply(clean_gene_name)

        # Create gene mapping
        gene_to_num, num_to_gene = create_gene_mapping(frame)

        # Convert to numbers
        frame_num = convert_frame_to_numbers(frame, gene_to_num)

        # Update nodesnum assignment
        nodesnum = args.nodes

        # Create results directory using the output_dir
        results_dir = os.path.join(args.output_dir, str(nodesnum))
        os.makedirs(results_dir, exist_ok=True)

        # Save network
        if nodesnum == 0:
            output_file = os.path.join(results_dir, "network_without_0.txt")
            frame_num.reset_index().to_csv(output_file, sep="\t", index=False)
            print(f"Saved network to: {output_file}")
        else:
            main(nodesnum, len(frame))

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
