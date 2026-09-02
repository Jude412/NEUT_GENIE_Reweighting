"""This code initializes the analysis by importing the files to reweight and writing the 
    training, validation and test parquet files, partitioned by topology, for the original
    and target distribution. The topologies with enough events are written to a json."""

# Imports 
from ROOT_file_conv import convert_input_file, topology_code
from Sample_io import split_sample, save_sample, TOPOLOGY_COLUMN
from Split_sizes import minimum_events
import argparse
import os
import json
import numpy as np

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to create the training, validation and test samples of every configured set of analysis parameters.")
    argparser.add_argument("--input_file_original", required=False, type=str, help="Path to the original input ROOT file.",
                           default="NEUT_test/test/t2knova.flattree.NEUT.ND280.CH.numu.BANFF_PRE.3411704_1.root")
    argparser.add_argument("--input_file_target", required=False, type=str, help="Path to the target input ROOT file.",
                           default="GENIE_test/test/t2knova.flattree.GENIE.ND280.CH.numu.2024NoGSF.3411716_1.root")
    argparser.add_argument("--input_tree_original", required=False, type=str, help="Name of the FlatTree in the original input file.",
                           default="T2KNOvATruthTree")
    argparser.add_argument("--input_tree_target", required=False, type=str, help="Name of the FlatTree in the target input file.",
                           default="T2KNOvATruthTree")
    argparser.add_argument("--analysis_params", nargs="+", help="List of parameters to extract from the branches for the analysis.")
    argparser.add_argument("--modes", type=int, nargs="+", help="Interaction modes to select (e.g. 1 for CCQE).",
                           required=False, default=[1])
    argparser.add_argument("--topologies", type=str, nargs="+", help="Interaction topologies to check sizes of, given by name (e.g. CC0pi).",
                           required=False, default=["CC0pi"])
    argparser.add_argument("--train_percentage", type=float, help="Percentage of the training sample (between 0 and 1).",
                           required=False, default=0.4)
    argparser.add_argument("--val_percentage", type=float, help="Percentage of the validation sample (between 0 and 1).",
                           required=False, default=0.4)
    argparser.add_argument("--random_seeds", type=int, nargs="+", required=False, 
                        help="List of 2 random seeds to use for the training and validation split.",default= [42, 43])
    argparser.add_argument("--output_dir", required=False, type=str,
                           help="Path to the output directory the samples will be saved in, each parameter set in a "
                                "sub-directory named after it.",
                           default="saved_samples/default/")
    argparser.add_argument("--topologies_file", required=False, type=str, help="Path to the output json file containing the topology codes to include", default="saved_samples/default/included_topologies.json")
    args = argparser.parse_args()

    # Getting data from the files
    print("Getting data from the files...")

    original = convert_input_file(args.input_file_original, args.input_tree_original, args.analysis_params, modes = args.modes)
    target = convert_input_file(args.input_file_target, args.input_tree_target, args.analysis_params, modes = args.modes)
    
    # Counting the topologies
    print(f"Counting topologies for sample {args.output_dir}...")

    orig_topology_counts = original[TOPOLOGY_COLUMN].value_counts().sort_index().to_dict()
    target_topology_counts = target[TOPOLOGY_COLUMN].value_counts().sort_index().to_dict()

    min_events = minimum_events(args.train_percentage, args.val_percentage)
    topologies_to_include = []

    for top in args.topologies:
        n_orig = orig_topology_counts.get(topology_code(top), 0)
        n_target = target_topology_counts.get(topology_code(top), 0)
        print(f"Topology {top}: {n_orig} original events, {n_target} target events.")

        if (n_orig < min_events) or (n_target < min_events):
            print(f"Topology {top} has too few events to be split into training, validation and test samples with the requested percentages. For this sample, the this topology will be skipped.")
            continue
        
        topologies_to_include.append(top)

    # Write out topologies_to_include as a json file
    os.makedirs(os.path.dirname(args.topologies_file), exist_ok=True)
    with open(args.topologies_file, 'w') as f:
        json.dump(topologies_to_include, f, indent=0)

    codes_to_include = [topology_code(top) for top in topologies_to_include]

    orig_train, orig_val, orig_test = split_sample(original, args.train_percentage, args.val_percentage, codes_to_include, args.random_seeds[0])
    target_train, target_val, target_test = split_sample(target, args.train_percentage, args.val_percentage, codes_to_include, args.random_seeds[1])

    # Saving the samples
    print(f"Saving the samples to {args.output_dir}...")
    save_sample(orig_train, os.path.join(args.output_dir, "original_train.parquet"))
    save_sample(orig_val, os.path.join(args.output_dir, "original_val.parquet"))
    save_sample(orig_test, os.path.join(args.output_dir, "original_test.parquet"))
    save_sample(target_train, os.path.join(args.output_dir, "target_train.parquet"))
    save_sample(target_val, os.path.join(args.output_dir, "target_val.parquet"))
    save_sample(target_test, os.path.join(args.output_dir, "target_test.parquet"))
    
