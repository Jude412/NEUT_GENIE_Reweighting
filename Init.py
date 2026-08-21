"""This code initializes the analysis by importing the files to reweight, creating the 3D and 8D samples, and computing the bootstrapped
SWD distribution for the target_test 3D and 8D samples. The sampels are saved in the "saved_samples + --output_dir" directory, and the SWD distribution is
saved in the "saved_swd_distribution + --output_dir" directory."""

# Imports 
from ROOT_file_conv import convert_input_file
from Sample_creation import create_samples
import numpy as np
import argparse
import os

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
    args = argparser.parse_args()

    # Getting data from the files
    print("Getting data from the files...")
    
    Index_8D_params = [args.analysis_params.index(param) for param in args.params_8D]
    Index_3D_params = [args.analysis_params.index(param) for param in args.params_3D]

    original = convert_input_file(args.input_file_original, args.input_tree_original, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies)
    target = convert_input_file(args.input_file_target, args.input_tree_target, args.branches, args.analysis_params, modes = args.modes, topologies = args.topologies)

    for name, dataset in (("original", original), ("target", target)):
        if len(dataset) < 2:
            raise ValueError(
                f"The {name} distribution holds {len(dataset)} event(s) for the topologies {args.topologies}. "
                "At least 2 events are needed to build the training, validation and test samples. "
                "Such sparse topologies are skipped automatically when running the whole workflow."
            )

    original_8D = original[:, Index_8D_params]
    target_8D = target[:, Index_8D_params]

    original_3D = original[:, Index_3D_params]
    target_3D = target[:, Index_3D_params]

    # Splitting data
    print("Splitting data into training, validation and test samples...")
    original_train, original_val, original_test = create_samples(original, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_train, target_val, target_test = create_samples(target, args.train_percentage, args.val_percentage, args.random_seeds[1])

    original_8D_train, original_8D_val, original_8D_test = create_samples(original_8D, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_8D_train, target_8D_val, target_8D_test = create_samples(target_8D, args.train_percentage, args.val_percentage, args.random_seeds[1])

    original_3D_train, original_3D_val, original_3D_test = create_samples(original_3D, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_3D_train, target_3D_val, target_3D_test = create_samples(target_3D, args.train_percentage, args.val_percentage, args.random_seeds[1])

    # We save the splitted samples as csv files
    os.makedirs(os.path.join(args.output_dir_samples_3D), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir_samples_8D), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir_samples_all), exist_ok=True)

    np.savetxt(os.path.join(args.output_dir_samples_all, "original_train.csv"), original_train, delimiter=",", header=",".join(args.analysis_params))
    np.savetxt(os.path.join(args.output_dir_samples_all, "original_val.csv"), original_val, delimiter=",", header=",".join(args.analysis_params))
    np.savetxt(os.path.join(args.output_dir_samples_all, "original_test.csv"), original_test, delimiter=",", header=",".join(args.analysis_params))
    np.savetxt(os.path.join(args.output_dir_samples_all, "target_train.csv"), target_train, delimiter=",", header=",".join(args.analysis_params))
    np.savetxt(os.path.join(args.output_dir_samples_all, "target_val.csv"), target_val, delimiter=",", header=",".join(args.analysis_params))
    np.savetxt(os.path.join(args.output_dir_samples_all, "target_test.csv"), target_test, delimiter=",", header=",".join(args.analysis_params))

    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_train.csv"), original_8D_train, delimiter=",", header=",".join(args.params_8D))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_val.csv"), original_8D_val, delimiter=",", header=",".join(args.params_8D))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_test.csv"), original_8D_test, delimiter=",", header=",".join(args.params_8D))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_train.csv"), target_8D_train, delimiter=",", header=",".join(args.params_8D))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_val.csv"), target_8D_val, delimiter=",", header=",".join(args.params_8D))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_test.csv"), target_8D_test, delimiter=",", header=",".join(args.params_8D))

    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_train.csv"), original_3D_train, delimiter=",", header=",".join(args.params_3D))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_val.csv"), original_3D_val, delimiter=",", header=",".join(args.params_3D))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_test.csv"), original_3D_test, delimiter=",", header=",".join(args.params_3D))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_train.csv"), target_3D_train, delimiter=",", header=",".join(args.params_3D))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_val.csv"), target_3D_val, delimiter=",", header=",".join(args.params_3D))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_test.csv"), target_3D_test, delimiter=",", header=",".join(args.params_3D))

    
