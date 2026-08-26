"""The goal of this script is to count, for a given sample, how many events fall into each of the
requested topologies in the original and in the target file. The counts are saved as a json file and
are used by the workflow to skip the topologies that are too sparse to be trained on."""

# imports
from ROOT_file_conv import convert_input_file, topology_code
import numpy as np
import argparse
import json
import os

def count_topologies(input_file, input_tree, branches, topologies, modes=None):
    """Return a dictionary giving, for each requested topology, the number of selected events in the file."""
    # "Topology" is the only parameter needed here, and no topology cut is applied so that every
    # requested topology can be counted in a single pass over the file.
    codes = convert_input_file(input_file, input_tree, branches, ["Topology"], modes=modes, topologies=None)[:, 0]
    return {str(topology): int(np.sum(codes == topology_code(topology))) for topology in topologies}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to count the number of events of each topology in the original and target files of a sample.")
    argparser.add_argument("--input_file_original", required=True, type=str, help="Path to the original input ROOT file.")
    argparser.add_argument("--input_file_target", required=True, type=str, help="Path to the target input ROOT file.")
    argparser.add_argument("--input_tree_original", required=False, type=str, help="Name of the FlatTree in the original input file.",
                           default="FlatTree_VARS")
    argparser.add_argument("--input_tree_target", required=False, type=str, help="Name of the FlatTree in the target input file.",
                           default="FlatTree_VARS")
    argparser.add_argument("--branches", nargs="+", help="List of branches to extract from the ROOT files.")
    argparser.add_argument("--modes", type=int, nargs="+", required=False, help="Interaction modes to select (e.g. 1 for CCQE).", default=[1])
    argparser.add_argument("--topologies", type=str, nargs="+", required=False, help="Interaction topologies to count, given by name (e.g. CC0pi).",
                           default=["CC0pi"])
    argparser.add_argument("--output_file", required=True, type=str, help="Path to the json file where the counts will be saved.")
    args = argparser.parse_args()

    print("Counting the events of each topology in the original and target files...")

    counts = {
        "original": count_topologies(args.input_file_original, args.input_tree_original, args.branches, args.topologies, modes=args.modes),
        "target": count_topologies(args.input_file_target, args.input_tree_target, args.branches, args.topologies, modes=args.modes),
    }

    for topology in args.topologies:
        print(f"  {topology}: {counts['original'][str(topology)]} original event(s), {counts['target'][str(topology)]} target event(s)")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(counts, f, indent=4)
