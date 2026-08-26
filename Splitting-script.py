"""The goal of this script is to read the files from GENIE/NEUT and to give the splitted
samples as output.
The user can choose the percentage of the training and validation samples, as well as the random seed for reproducibility. 
The output samples are numpy arrays that will be used to train the reweighting techniques and validate their performance.
They are saved as csv files in a specified directory. The script can be run from the command line with the appropriate arguments."""

#imports 
from ROOT_file_conv import convert_input_file
from Sample_io import save_sample, create_samples
import argparse
import os
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to create the training and validation samples for the reweighting techniques.")
    argparser.add_argument("--input_file_original", required=False, type=str, help="Path to the original input ROOT file.",
                           default="/vols/dune/jmm224/t2knova/reweighting/NEUT_files/T2KND_FHC_numu_H2O_NEUT562_1M_0000_NUISFLAT.root")
    argparser.add_argument("--input_file_target", required=False, type=str, help="Path to the target input ROOT file.",
                           default="/vols/dune/jmm224/t2knova/reweighting/GENIE_files/T2KND_FHC_numu_H2O_GENIEv3_G18_10b_00_000_1M_0000_NUISFLAT.root")
    argparser.add_argument("--input_tree_original", required=False, type=str, help="Name of the FlatTree in the original input file.",
                           default="FlatTree_VARS")
    argparser.add_argument("--input_tree_target", required=False, type=str, help="Name of the FlatTree in the target input file.",
                           default="FlatTree_VARS")
    argparser.add_argument("--split_indices", required=True, type=str, help="Path to the json file containing the indices for training, validation and test samples.")
    argparser.add_argument("--branches", nargs="+", help="List of branches to extract from the ROOT files.")
    argparser.add_argument("--analysis_params", nargs="+", help="List of parameters to extract from the branches for the analysis.")
    argparser.add_argument("--modes", type=int, nargs="+", required=False, help="Interaction modes to select (e.g. 1 for CCQE).", default=[1] )
    argparser.add_argument("--topologies", type=str, nargs="+", required=False, help="Interaction topologies to select, given by name (e.g. CC0pi).", default=["CC0pi"] )
    argparser.add_argument("--samples_dir", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/")
    argparser.add_argument("--parameters_interest", nargs='+', required=False, help="List of parameters to keep from the original files, in the format 'param1,param2,...'.",
                           default=["Enu_true", "ELep", "CosLep", "W", "Eav"])
    args = argparser.parse_args()

    # Getting data from the files
    print("Getting data from the files...")

    Params_of_interest = args.parameters_interest

    original_load, original_weights = convert_input_file(args.input_file_original, args.input_tree_original, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies, return_weights = True)
    target_load, target_weights = convert_input_file(args.input_file_target, args.input_tree_target, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies, return_weights = True)

    original = original_load[:, [args.analysis_params.index(param) for param in args.parameters_interest]]
    target = target_load[:, [args.analysis_params.index(param) for param in args.parameters_interest]]

    # We split the data into a training and validation set
    # The weights follow their events, so that every sample holds the weights of the events it contains.
    print("Splitting data into training, validation and test samples...")
    with open(args.split_indices, 'r') as f:
        split_indices = json.load(f)

    original_train, original_val, original_test, original_train_w, original_val_w, original_test_w = create_samples(original, split_indices["original"], weights=original_weights)
    target_train, target_val, target_test, target_train_w, target_val_w, target_test_w = create_samples(target, split_indices["target"], weights=target_weights)

    # We save the splitted samples as csv files
    os.makedirs(args.samples_dir, exist_ok=True)
    save_sample(os.path.join(args.samples_dir, "original_train.csv"), original_train, original_train_w, args.parameters_interest)
    save_sample(os.path.join(args.samples_dir, "original_val.csv"), original_val, original_val_w, args.parameters_interest)
    save_sample(os.path.join(args.samples_dir, "original_test.csv"), original_test, original_test_w, args.parameters_interest)
    save_sample(os.path.join(args.samples_dir, "target_train.csv"), target_train, target_train_w, args.parameters_interest)
    save_sample(os.path.join(args.samples_dir, "target_val.csv"), target_val, target_val_w, args.parameters_interest)
    save_sample(os.path.join(args.samples_dir, "target_test.csv"), target_test, target_test_w, args.parameters_interest)

