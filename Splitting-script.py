"""The goal of this script is to read the files from GENIE/NEUT and to give the splitted
samples as output.
The user can choose the percentage of the training and validation samples, as well as the random seed for reproducibility. 
The output samples are numpy arrays that will be used to train the reweighting techniques and validate their performance.
They are saved as csv files in a specified directory. The script can be run from the command line with the appropriate arguments."""

#imports 
from ROOT_file_conv import convert_input_file
from Sample_creation import create_samples
import numpy as np
import argparse
import os

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
    argparser.add_argument("--branches", nargs="+", help="List of branches to extract from the ROOT files.")
    argparser.add_argument("--analysis_params", nargs="+", help="List of parameters to extract from the branches for the analysis.")
    argparser.add_argument("--modes", type=int, nargs="+", required=False, help="Interaction modes to select (e.g. 1 for CCQE).", default=[1] )
    argparser.add_argument("--topologies", type=int, nargs="+", required=False, help="Interaction topologies to select (e.g. 1 for CCQE).", default=[1] )
    # argparser.add_argument("--neutrino_PDG", type=int, required=False, help="PDG code of the neutrino type to select (e.g. 14 for numu).", default = 14)
    argparser.add_argument("--train_percentage", type=float, required=False, help="Percentage of the training sample (between 0 and 1).", default=0.4)
    argparser.add_argument("--val_percentage", type=float, required=False, help="Percentage of the validation sample (between 0 and 1).", default=0.4)
    argparser.add_argument("--random_seeds", type = list, required=False, 
                        help="List of 2 random seeds to use for the training and validation split.", default= [42, 43])
    argparser.add_argument("--samples_dir", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/")
    argparser.add_argument("--parameters_interest", nargs='+', required=False, help="List of parameters to keep from the original files, in the format 'param1,param2,...'.",
                           default=["Enu_true", "ELep", "CosLep", "W", "Eav"])
    args = argparser.parse_args()

    # Getting data from the files
    print("Getting data from the files...")

    Params_of_interest = args.parameters_interest

    original_load = convert_input_file(args.input_file_original, args.input_tree_original, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies)
    target_load = convert_input_file(args.input_file_target, args.input_tree_target, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies)

    original = original_load[:, [args.analysis_params.index(param) for param in args.parameters_interest]]
    target = target_load[:, [args.analysis_params.index(param) for param in args.parameters_interest]]

    # We split the data into a training and validation set
    print("Splitting data into training, validation and test samples...")
    original_train, original_val, original_test = create_samples(original, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_train, target_val, target_test = create_samples(target, args.train_percentage, args.val_percentage, args.random_seeds[1])

    # We save the splitted samples as csv files
    os.makedirs(args.samples_dir, exist_ok=True)
    np.savetxt(os.path.join(args.samples_dir, "original_train.csv"), original_train, delimiter=",", header=",".join(args.parameters_interest))
    np.savetxt(os.path.join(args.samples_dir, "original_val.csv"), original_val, delimiter=",", header=",".join(args.parameters_interest))
    np.savetxt(os.path.join(args.samples_dir, "original_test.csv"), original_test, delimiter=",", header=",".join(args.parameters_interest))
    np.savetxt(os.path.join(args.samples_dir, "target_train.csv"), target_train, delimiter=",", header=",".join(args.parameters_interest))
    np.savetxt(os.path.join(args.samples_dir, "target_val.csv"), target_val, delimiter=",", header=",".join(args.parameters_interest))
    np.savetxt(os.path.join(args.samples_dir, "target_test.csv"), target_test, delimiter=",", header=",".join(args.parameters_interest))

