"""This code initializes the analysis by importing the files to reweight and creating the samples of every
configured set of analysis parameters. The events are read once and split into a training, a validation and
a test sample, which are then written for every parameter set in the "--output_dir" directory, in a
sub-directory named after the set. The indices of the split are saved next to them."""

# Imports 
from ROOT_file_conv import convert_input_file
from Sample_io import save_sample, create_samples
from Param_sets import ALL_PARAMS_SET, add_param_set_argument, param_sets_from_args, indices_of
from Split_sizes import minimum_events
import argparse
import os
import json
import numpy as np

# The samples the events of a distribution are split into.
SPLITS = ("train", "val", "test")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to create the training, validation and test samples of every configured set of analysis parameters.")
    argparser.add_argument("--input_file_original", required=False, type=str, help="Path to the original input ROOT file.",
                           default="/vols/dune/jmm224/t2knova/reweighting/NEUT_files/T2KND_FHC_numu_H2O_NEUT562_1M_0000_NUISFLAT.root")
    argparser.add_argument("--input_file_target", required=False, type=str, help="Path to the target input ROOT file.",
                           default="/vols/dune/jmm224/t2knova/reweighting/GENIE_files/T2KND_FHC_numu_H2O_GENIEv3_G18_10b_00_000_1M_0000_NUISFLAT.root")
    argparser.add_argument("--input_tree_original", required=False, type=str, help="Name of the FlatTree in the original input file.",
                           default="FlatTree_VARS")
    argparser.add_argument("--input_tree_target", required=False, type=str, help="Name of the FlatTree in the target input file.",
                           default="FlatTree_VARS")
    argparser.add_argument("--branches", nargs="+", help="List of branches to extract from the ROOT files.")
    argparser.add_argument("--analysis_params", nargs="+", help="List of parameters to extract from the branches for the analysis.")
    add_param_set_argument(argparser, help_suffix=f" The '{ALL_PARAMS_SET}' set holding every analysis parameter is always written.")
    argparser.add_argument("--modes", type=int, nargs="+", help="Interaction modes to select (e.g. 1 for CCQE).",
                           required=False, default=[1])
    argparser.add_argument("--topologies", type=str, nargs="+", help="Interaction topologies to select, given by name (e.g. CC0pi).",
                           required=False, default=["CC0pi"])
    # argparser.add_argument("--neutrino_PDG", type=int, help="PDG code of the neutrino type to select (e.g. 14 for numu).", 
    #                        required=False, default = 14)
    argparser.add_argument("--train_percentage", type=float, help="Percentage of the training sample (between 0 and 1).",
                           required=False, default=0.4)
    argparser.add_argument("--val_percentage", type=float, help="Percentage of the validation sample (between 0 and 1).",
                           required=False, default=0.4)
    argparser.add_argument("--random_seeds", type=int, nargs="+", required=False, 
                        help="List of 2 random seeds to use for the training and validation split.",default= [42, 43])
    argparser.add_argument("--output_dir", required=False, type=str,
                           help="Path to the output directory the samples will be saved in, each parameter set in a "
                                "sub-directory named after it.",
                           default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/")
    argparser.add_argument("--indices_file", required=False, type=str, help="Path to the output json file containing the indices for training, validation and test samples.", default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/split_indices.json")
    args = argparser.parse_args()

    # The set holding every analysis parameter is always written, as the metrics of the other sets
    # are computed from the samples holding it.
    param_sets = {ALL_PARAMS_SET: list(args.analysis_params)}
    param_sets.update(param_sets_from_args(args.param_set))
    set_indices = {name: indices_of(params, args.analysis_params) for name, params in param_sets.items()}

    # Getting data from the files
    print("Getting data from the files...")

    original, original_weights = convert_input_file(args.input_file_original, args.input_tree_original, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies, return_weights = True)
    target, target_weights = convert_input_file(args.input_file_target, args.input_tree_target, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies, return_weights = True)

    min_events = minimum_events(args.train_percentage, args.val_percentage)
    for name, dataset in (("original", original), ("target", target)):
        if len(dataset) < min_events:
            raise ValueError(
                f"The {name} distribution holds {len(dataset)} event(s) for the topologies {args.topologies}. "
                f"At least {min_events} events are needed to build the training, validation and test samples "
                f"with a training percentage of {args.train_percentage} and a validation percentage of "
                f"{args.val_percentage}. "
                "Such sparse topologies are skipped automatically when running the whole workflow."
            )

    # Splitting data
    print("Splitting data into training, validation and test samples...")

    split_indices = {
            "original": {split: [] for split in SPLITS},
            "target": {split: [] for split in SPLITS}
            }
    for idx, (name, distribution) in enumerate([("original", original), ("target", target)]):
        indices = np.arange(distribution.shape[0])
        np.random.seed(args.random_seeds[idx])
        split_indices[name]["train"] = np.random.choice(indices, size=int(args.train_percentage*len(distribution)), replace=False).tolist()
        all_but_train_idx = np.setdiff1d(indices, split_indices[name]["train"])
        split_indices[name]["val"] = np.random.choice(all_but_train_idx, size=int(args.val_percentage*len(distribution)), replace=False).tolist()
        split_indices[name]["test"] = np.setdiff1d(all_but_train_idx, split_indices[name]["val"]).tolist()

    # Save the split indices to a JSON file
    indices_dir = os.path.dirname(args.indices_file)
    if indices_dir:
        os.makedirs(indices_dir, exist_ok=True)
    with open(args.indices_file, "w") as f:
        json.dump(split_indices, f, indent=0)

    # We save the splitted samples of every parameter set as csv files. The events are split once,
    # so that a given sample holds the same events in every parameter set.
    samples = {
        "original": create_samples(original, split_indices["original"], weights=original_weights),
        "target": create_samples(target, split_indices["target"], weights=target_weights),
    }

    for set_name, params in param_sets.items():
        set_dir = os.path.join(args.output_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)

        for distribution_name, (split_samples, split_weights) in samples.items():
            for split in SPLITS:
                save_sample(os.path.join(set_dir, f"{distribution_name}_{split}.csv"),
                            split_samples[split][:, set_indices[set_name]], split_weights[split], params)

        print(f"Wrote the samples of the parameter set '{set_name}' ({params}) in {set_dir}.")
