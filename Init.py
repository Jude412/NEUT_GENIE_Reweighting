"""This code initializes the analysis by importing the files to reweight, creating the 3D and 8D samples, and computing the bootstrapped
SWD distribution for the target_test 3D and 8D samples. The sampels are saved in the "saved_samples + --output_dir" directory, and the SWD distribution is
saved in the "saved_swd_distribution + --output_dir" directory."""

# Imports 
from NEUTFile_conv_ndim import convert_NEUT_input_file_ndim, convert_NEUT_input_file_alldim
from GENIEFile_conv_ndim import convert_GENIE_input_file_ndim, convert_GENIE_input_file_alldim
from Sample_creation import create_samples
import numpy as np
import argparse
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to create the 3D and 8D training and validation samples and bootstrappedSWD distribution for the reweighting techniques.")
    argparser.add_argument("--input_file_NEUT", required=False, type=str, help="Path to the input ROOT file from Nuisance (that used NEUT as generator).",
                           default="/vols/dune/jmm224/t2knova/reweighting/NEUT_files/T2KND_FHC_numu_H2O_NEUT562_1M_0000_NUISFLAT.root")
    argparser.add_argument("--input_file_GENIE", required=False, type=str, help="Path to the input ROOT file from Nuisance (that used GENIE as generator).",
                           default="/vols/dune/jmm224/t2knova/reweighting/GENIE_files/T2KND_FHC_numu_H2O_GENIEv3_G18_10b_00_000_1M_0000_NUISFLAT.root")
    argparser.add_argument("--modes", type=int, nargs="+", help="Interaction modes to select (e.g. 1 for CCQE).",
                           required=False, default=[1])
    argparser.add_argument("--modes_v2", type=int, nargs="+", help="Interaction modes v2 to select (e.g. 1 for CC0pi).",
                           required=False, default=[1])
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
    argparser.add_argument("--output_dir_samples_21D", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/21D/")
    args = argparser.parse_args()

    # Getting data from the files
    print("Getting data from the files...")
    # The params we want from the files
    List_parameters_analysis = ["Enu_true", "Plep", "CosLep", "Q2", "q0", "q3", "PTlep", "Eav", "W", "y", "Mode", "Mode_v2",
                "cc", "hitnuc", "N_n", "K_n", "N_p", "K_p", "N_pi0", "K_pi0", "N_pip", "K_pip", "N_pim", "K_pim"]
    # The output from reading the files
    List_all_params = ["Enu_true", "Plep", "CosLep", "Q2", "q0", "q3", "W", "Eav", "y", "PTlep",
                 "PDGnu", "Mode", "Mode_v2", "cc", "hitnuc", "A", "N_n", "K_n", "N_p", "K_p", "N_pi0", "K_pi0", "N_pip", "K_pip", "N_pim", "K_pim"]
    
    Index_21D_params = [List_all_params.index(param) for param in List_parameters_analysis]
    Index_8D_params = [List_all_params.index(param) for param in List_parameters_analysis[:8]]
    Index_3D_params = [List_all_params.index(param) for param in List_parameters_analysis[:3]]

    original = convert_NEUT_input_file_alldim(args.input_file_NEUT, modes = args.modes, modes_v2 = args.modes_v2)
    target = convert_GENIE_input_file_alldim(args.input_file_GENIE, modes = args.modes, modes_v2 = args.modes_v2)

    original_21D = original[:, Index_21D_params]
    target_21D = target[:, Index_21D_params]

    original_8D = original[:, Index_8D_params]
    target_8D = target[:, Index_8D_params]

    original_3D = original[:, Index_3D_params]
    target_3D = target[:, Index_3D_params]

    # Splitting data
    print("Splitting data into training, validation and test samples...")
    original_train, original_val, original_test = create_samples(original_21D, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_train, target_val, target_test = create_samples(target_21D, args.train_percentage, args.val_percentage, args.random_seeds[1])

    original_8D_train, original_8D_val, original_8D_test = create_samples(original_8D, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_8D_train, target_8D_val, target_8D_test = create_samples(target_8D, args.train_percentage, args.val_percentage, args.random_seeds[1])

    original_3D_train, original_3D_val, original_3D_test = create_samples(original_3D, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_3D_train, target_3D_val, target_3D_test = create_samples(target_3D, args.train_percentage, args.val_percentage, args.random_seeds[1])

    # We save the splitted samples as csv files
    os.makedirs(os.path.join(args.output_dir_samples_3D), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir_samples_8D), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir_samples_21D), exist_ok=True)

    np.savetxt(os.path.join(args.output_dir_samples_21D, "original_train.csv"), original_train, delimiter=",", header=",".join(List_parameters_analysis))
    np.savetxt(os.path.join(args.output_dir_samples_21D, "original_val.csv"), original_val, delimiter=",", header=",".join(List_parameters_analysis))
    np.savetxt(os.path.join(args.output_dir_samples_21D, "original_test.csv"), original_test, delimiter=",", header=",".join(List_parameters_analysis))
    np.savetxt(os.path.join(args.output_dir_samples_21D, "target_train.csv"), target_train, delimiter=",", header=",".join(List_parameters_analysis))
    np.savetxt(os.path.join(args.output_dir_samples_21D, "target_val.csv"), target_val, delimiter=",", header=",".join(List_parameters_analysis))
    np.savetxt(os.path.join(args.output_dir_samples_21D, "target_test.csv"), target_test, delimiter=",", header=",".join(List_parameters_analysis))

    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_train.csv"), original_8D_train, delimiter=",", header=",".join(List_parameters_analysis[:8]))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_val.csv"), original_8D_val, delimiter=",", header=",".join(List_parameters_analysis[:8]))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_test.csv"), original_8D_test, delimiter=",", header=",".join(List_parameters_analysis[:8]))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_train.csv"), target_8D_train, delimiter=",", header=",".join(List_parameters_analysis[:8]))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_val.csv"), target_8D_val, delimiter=",", header=",".join(List_parameters_analysis[:8]))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_test.csv"), target_8D_test, delimiter=",", header=",".join(List_parameters_analysis[:8]))

    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_train.csv"), original_3D_train, delimiter=",", header=",".join(List_parameters_analysis[:3]))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_val.csv"), original_3D_val, delimiter=",", header=",".join(List_parameters_analysis[:3]))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_test.csv"), original_3D_test, delimiter=",", header=",".join(List_parameters_analysis[:3]))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_train.csv"), target_3D_train, delimiter=",", header=",".join(List_parameters_analysis[:3]))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_val.csv"), target_3D_val, delimiter=",", header=",".join(List_parameters_analysis[:3]))
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_test.csv"), target_3D_test, delimiter=",", header=",".join(List_parameters_analysis[:3]))

    
