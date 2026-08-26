"""This code initializes the analysis by importing the files to reweight, creating the 3D and 8D samples, and computing the bootstrapped
SWD distribution for the target_test 3D and 8D samples. The sampels are saved in the "saved_samples + --output_dir" directory, and the SWD distribution is
saved in the "saved_swd_distribution + --output_dir" directory."""

# Imports 
from ROOT_file_conv import convert_input_file
from Sample_io import save_sample, create_samples
from split_sizes import minimum_events
import argparse
import os
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to create the 3D and 8D training and validation samples and bootstrappedSWD distribution for the reweighting techniques.")
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
    argparser.add_argument("--params_8D", nargs="+", help="List of parameters to use in 8D samples.")
    argparser.add_argument("--params_3D", nargs="+", help="List of parameters to use in 3D samples.")
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
    argparser.add_argument("--random_seeds", type = list, required=False, 
                        help="List of 2 random seeds to use for the training and validation split.",default= [42, 43])
    argparser.add_argument("--output_dir_samples_3D", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/3D/")
    argparser.add_argument("--output_dir_samples_8D", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/8D/")
    argparser.add_argument("--output_dir_samples_all", "--output_dir_samples_21D", dest="output_dir_samples_all", required=False, type=str, help="Path to the output directory where the samples containing all configured analysis parameters will be saved.",
                            default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/all/")
    argparser.add_argument("--indices_file", required=False, type=str, help="Path to the output json file containing the indices for training, validation and test samples.", default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/split_indices.json")
    args = argparser.parse_args()

    # Getting data from the files
    print("Getting data from the files...")
    
    Index_8D_params = [args.analysis_params.index(param) for param in args.params_8D]
    Index_3D_params = [args.analysis_params.index(param) for param in args.params_3D]

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

    original_8D = original[:, Index_8D_params]
    target_8D = target[:, Index_8D_params]

    original_3D = original[:, Index_3D_params]
    target_3D = target[:, Index_3D_params]

    # Splitting data
    print("Splitting data into training, validation and test samples...")

    split_indices = {
            "original": {"train": [], "val": [], "test": []},
            "target": {"train": [], "val": [], "test": []}
            }
    for idx, (name, distribution) in enumerate([("original", original), ("target", target)]):
        indices = np.arange(distribution.shape[0])
        np.random.seed(args.random_seeds[idx])
        split_indices[name]["train"] = np.random.choice(indices, size=int(percentage_train*len(distribution)), replace=False)
        all_but_train_idx = np.setdiff1d(indices, train_indices[name])
        split_indices[name]["val"] = np.random.choice(all_but_train_idx, size=int(percentage_val*len(distribution)), replace=False)
        split_indices[name]["test"] = np.setdiff1d(all_but_train_idx, val_indices)


    original_train, original_val, original_test, original_train_w, original_val_w, original_test_w = create_samples(original, split_indices["original"], weights=original_weights)
    target_train, target_val, target_test, target_train_w, target_val_w, target_test_w = create_samples(target, split_indices["target"], weights=target_weights)

    original_8D_train, original_8D_val, original_8D_test = create_samples(original_8D, split_indices["original"])
    target_8D_train, target_8D_val, target_8D_test = create_samples(target_8D, split_indices["target"])

    original_3D_train, original_3D_val, original_3D_test = create_samples(original_3D, split_indices["original"])
    target_3D_train, target_3D_val, target_3D_test = create_samples(target_3D, split_indices["target"])

    # We save the splitted samples as csv files
    os.makedirs(os.path.join(args.output_dir_samples_3D), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir_samples_8D), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir_samples_all), exist_ok=True)

    # Save the split indices to a JSON file
    with open(args.indices_file, "w") as f:
        json.dump(split_indices, f)

    save_sample(os.path.join(args.output_dir_samples_all, "original_train.csv"), original_train, original_train_w, args.analysis_params)
    save_sample(os.path.join(args.output_dir_samples_all, "original_val.csv"), original_val, original_val_w, args.analysis_params)
    save_sample(os.path.join(args.output_dir_samples_all, "original_test.csv"), original_test, original_test_w, args.analysis_params)
    save_sample(os.path.join(args.output_dir_samples_all, "target_train.csv"), target_train, target_train_w, args.analysis_params)
    save_sample(os.path.join(args.output_dir_samples_all, "target_val.csv"), target_val, target_val_w, args.analysis_params)
    save_sample(os.path.join(args.output_dir_samples_all, "target_test.csv"), target_test, target_test_w, args.analysis_params)

    save_sample(os.path.join(args.output_dir_samples_8D, "original_train.csv"), original_8D_train, original_train_w, args.params_8D)
    save_sample(os.path.join(args.output_dir_samples_8D, "original_val.csv"), original_8D_val, original_val_w, args.params_8D)
    save_sample(os.path.join(args.output_dir_samples_8D, "original_test.csv"), original_8D_test, original_test_w, args.params_8D)
    save_sample(os.path.join(args.output_dir_samples_8D, "target_train.csv"), target_8D_train, target_train_w, args.params_8D)
    save_sample(os.path.join(args.output_dir_samples_8D, "target_val.csv"), target_8D_val, target_val_w, args.params_8D)
    save_sample(os.path.join(args.output_dir_samples_8D, "target_test.csv"), target_8D_test, target_test_w, args.params_8D)

    save_sample(os.path.join(args.output_dir_samples_3D, "original_train.csv"), original_3D_train, original_train_w, args.params_3D)
    save_sample(os.path.join(args.output_dir_samples_3D, "original_val.csv"), original_3D_val, original_val_w, args.params_3D)
    save_sample(os.path.join(args.output_dir_samples_3D, "original_test.csv"), original_3D_test, original_test_w, args.params_3D)
    save_sample(os.path.join(args.output_dir_samples_3D, "target_train.csv"), target_3D_train, target_train_w, args.params_3D)
    save_sample(os.path.join(args.output_dir_samples_3D, "target_val.csv"), target_3D_val, target_val_w, args.params_3D)
    save_sample(os.path.join(args.output_dir_samples_3D, "target_test.csv"), target_3D_test, target_test_w, args.params_3D)
