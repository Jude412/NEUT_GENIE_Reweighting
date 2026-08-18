"""The goal of this script is to evaluate the performance of the different reweighting methods. 
The idea is to have functions that take as input the original and target distributions, and the predicted weights of any
method and return : 1D and 2D histograms with ratio of reweighting data over target, a Chi2 statistics, a SWD value and the associated p-value.
We are here trying to implement it with n-dimensional distributions, for which we assume the weights have already been predicted."""

#imports
import os

from Metrics_ndim import chi2_hist_axis, plot_histograms, plot_2D_histogram, chi2_dof, compute_swd, compute_p_value, chi2_p_value
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Evaluate the performance of reweighting methods.')
    argparser.add_argument('--original_test', type=str, required=False, help='Path to the original 21D test dataset (csv).',
                            default='/vols/dune/jmm224/t2knova/reweighting/saved_samples/original_test_21D.csv')
    argparser.add_argument('--target_test', type=str, required=False, help='Path to the target 21D test dataset (csv).',
                            default='/vols/dune/jmm224/t2knova/reweighting/saved_samples/target_test_21D.csv')
    argparser.add_argument('--weights_paths', type=str, required=False,
                           help = "path to the json file containing a dictionnary with each method and path to its predicted weights (csv).",
                           default='/vols/dune/jmm224/t2knova/reweighting/make_metrics.json')
    argparser.add_argument('--output_file', type=str, required=False, help='File to save the output plots and metrics.',
                            default='/vols/dune/jmm224/t2knova/reweighting/saved_figures/histograms_and_ratios')
    argparser.add_argument('--make_1D_plots', action=argparse.BooleanOptionalAction, help='Whether to make the 1D plots or not.')
    argparser.add_argument('--make_2D_plots', action=argparse.BooleanOptionalAction, help='Whether to make the 2D plots or not.')
    argparser.add_argument('--compute_chi2', action=argparse.BooleanOptionalAction, help='Whether to compute the Chi2 statistic or not.')
    argparser.add_argument('--compute_swd', action=argparse.BooleanOptionalAction, help='Whether to compute the SWD metric or not.')
    argparser.add_argument('--interest_params', nargs='+', required=False, help='Parameters used for the custom SWD distribution, in the format "param1,param2,...".',
                            default = ["Enu_true", "Plep", "CosLep"])
    argparser.add_argument('--custom_swd_distribution', type=str, required=False, help='Path to a custom SWD distribution to compute the p-value with.')
    argparser.add_argument("--swd_distribution_3D", type=str, required=False, help="Path to the SWD distribution file or to store it.", 
                        default = "/vols/dune/jmm224/t2knova/reweighting/swd_distribution/list_swd_3D_test20p.npy")
    argparser.add_argument("--swd_distribution_8D", type=str, required=False, help="Path to the SWD distribution file or to store it.",
                            default = "/vols/dune/jmm224/t2knova/reweighting/swd_distribution/list_swd_8D_test20p.npy")
    argparser.add_argument("--swd_distribution_21D", type=str, required=False, help="Path to the SWD distribution file or to store it.",
                            default = "/vols/dune/jmm224/t2knova/reweighting/swd_distribution/list_swd_21D_test20p.npy")
    argparser.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    argparser.add_argument("--binning_file", type=str, required=False, help="Path to the json file containing the binning information for each parameter.",
                           default="/vols/dune/jmm224/t2knova/reweighting/binnings.json")
    args = argparser.parse_args()

    List_all_parameters = ["Enu_true", "Plep", "CosLep", "Q2", "q0", "q3", "PTlep", "Eav", "W", "y", "Mode", "Mode_v2",
                "cc", "hitnuc", "N_n", "K_n", "N_p", "K_p", "N_pi0", "K_pi0", "N_pip", "K_pip", "N_pim", "K_pim"]
    
    List_labels_all_parameters = [r"$E^{true}_{\nu}$ (GeV)", r"$p_{lep}$ (GeV/c)", r"$cos(\theta_{lep})$",
                                r"$Q^2$ (GeV$^2$/c$^2$)", r"$q_0$ (GeV)", r"$q_3$ (GeV/c)", r"$p^T_{lep}$ (GeV/c)",
                                r"$E_{Av}$ (GeV)", r"$W$ (GeV/c$^2$)", "y", "Mode", "Mode_v2",
                                "cc", "hitnuc", r"$N_n$", r"$K_n$ (GeV)", r"$N_p$", r"$K_p$ (GeV)", r"$N_{pi^0}$", r"$K_{pi^0}$ (GeV)", r"$N_{pi^+}$", r"$K_{pi^+}$ (GeV)",
                                r"$N_{pi^-}$", r"$K_{pi^-}$ (GeV)"]
    
    Index_parameters = [List_all_parameters.index(param) for param in args.interest_params]

    # We load the data and the weights
    original_test = np.loadtxt(args.original_test, delimiter=",")
    target_test = np.loadtxt(args.target_test, delimiter=",")

    with open(args.weights_paths, 'r') as f:
        weights_path_dict = json.load(f)
    
    weights_dict = {}
    for method, path in weights_path_dict.items():
        weights_dict[method] = np.loadtxt(path, delimiter=",")
    
    weights_dict['Original'] = np.ones(len(original_test))/len(original_test)
    
    # Make the plots

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    if args.make_1D_plots:
        plot_histograms(original_test, target_test, weights_dict,
                        dict_binning=binning_dict,
                        xlabels = List_labels_all_parameters,
                        variables = List_all_parameters, 
                        output_file=os.path.join(args.output_file, "1Dhist.pdf"))
    
    if args.make_2D_plots:
        plot_2D_histogram(original_test, target_test, weights_dict, 
                        xlabels = args.interest_params,
                        nbins = 30, 
                        pull = True,
                        output_file=os.path.join(args.output_file, "2Dhist.pdf"))
    
    # Compute the metrics
        
    if args.compute_chi2:
        chi2_dict = {}
        for key in weights_dict.keys():
            chi2_dim = {}
            for param in List_all_parameters:
                param_index = List_all_parameters.index(param)
                if param in binning_dict.keys():
                    x_min, x_max = binning_dict[param]["x_min"], binning_dict[param]["x_max"]
                    n_bins = binning_dict[param]["n_bins"]
                else:
                    x_min, x_max = None, None
                    n_bins = 30
                chi2_val, dof = chi2_hist_axis(original_test, target_test, weights_dict[key], param_index, n_bins=n_bins, x_min=x_min, x_max=x_max)  
                chi2_dim[param] = chi2_val/dof if dof > 0 else 0
                chi2_dim[param+"_p_value"] = chi2_p_value(chi2_val, dof)
            chi2_dict[key] = chi2_dim

        chi2_3D_dict = chi2_dof(original_test[:, :3], target_test[:, :3], weights_dict, binning_dict=binning_dict, List_param_interest = List_all_parameters[:3])
        chi2_3D_p_values = {key: chi2_p_value(chi2*87, 87) for key, chi2 in chi2_3D_dict.items()}
        chi2_8D_dict = chi2_dof(original_test[:, :8], target_test[:, :8], weights_dict, binning_dict=binning_dict, List_param_interest = List_all_parameters[:8])
        chi2_21D_dict = chi2_dof(original_test, target_test, weights_dict, binning_dict=binning_dict, List_param_interest = List_all_parameters)


    if args.compute_swd:
        if args.custom_swd_distribution is not None:
            swd_custom_list = np.load(args.custom_swd_distribution)
            swd_dict_custom = compute_swd(original_test[:, Index_parameters], target_test[:, Index_parameters], weights_dict, n_directions=args.n_directions)
            p_value_dict_custom = compute_p_value(swd_dict_custom, swd_custom_list)

        swd_list_3D = np.load(args.swd_distribution_3D)
        swd_list_8D = np.load(args.swd_distribution_8D)
        swd_list_21D = np.load(args.swd_distribution_21D)
        swd_dict_3D = compute_swd(original_test[:, :3], target_test[:, :3], weights_dict, n_directions=args.n_directions)
        swd_dict_8D = compute_swd(original_test[:, :8], target_test[:, :8], weights_dict, n_directions=args.n_directions)
        swd_dict_21D = compute_swd(original_test, target_test, weights_dict, n_directions=args.n_directions)

        p_value_dict_3D = compute_p_value(swd_dict_3D, swd_list_3D)
        p_value_dict_8D = compute_p_value(swd_dict_8D, swd_list_8D)
        p_value_dict_21D = compute_p_value(swd_dict_21D, swd_list_21D)

    # Save the metrics in a json file
    metrics_dict = {}

    if args.compute_chi2:
        metrics_dict['chi2_1D'] = chi2_dict
        metrics_dict['chi2_3D'] = chi2_3D_dict
        metrics_dict['chi2_3D_p_value'] = chi2_3D_p_values
        metrics_dict['chi2_8D'] = chi2_8D_dict
        metrics_dict['chi2_21D'] = chi2_21D_dict
    if args.compute_swd:
        if args.custom_swd_distribution is not None:
            metrics_dict['swd_custom'] = swd_dict_custom
            metrics_dict['p_value_custom'] = p_value_dict_custom

        metrics_dict['swd_3D'] = swd_dict_3D
        metrics_dict['swd_8D'] = swd_dict_8D
        metrics_dict['swd_21D'] = swd_dict_21D
        metrics_dict['p_value_3D'] = p_value_dict_3D
        metrics_dict['p_value_8D'] = p_value_dict_8D
        metrics_dict['p_value_21D'] = p_value_dict_21D
    with open(os.path.join(args.output_file, "metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=0)

    # Plot the training history if possible

    if "NN" in weights_path_dict.keys():
        training_history = np.loadtxt(weights_path_dict["NN"].replace("saved_weights", "saved_models").replace("Training_4D", "4D").replace("weights_test", "training_history"), delimiter=',')
        with PdfPages(os.path.join(args.output_file, "NN_training_history.pdf")) as pdf:
            plt.plot(training_history[:len(training_history)//2], label='Training Loss')
            plt.plot(training_history[len(training_history)//2:], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training History for NN {len(Index_parameters)}D training')
            plt.legend()
            pdf.savefig()
            plt.close()

    if "XGB" in weights_path_dict.keys():
        model_path = weights_path_dict["XGB"].replace("saved_weights", "saved_models").replace("Training_4D", "4D").replace("weights_test", "model").replace("csv", "pkl")
        model = pickle.load(open(model_path, "rb"))
        training_history = model.evals_result()
        with PdfPages(os.path.join(args.output_file, "XGB_training_history.pdf")) as pdf:
            plt.plot(training_history["validation_0"]["logloss"], label='Train Log Loss')
            plt.plot(training_history["validation_1"]["logloss"], label='Validation Log Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            plt.title(f'Training History for XGB {len(Index_parameters)}D training')
            plt.legend()
            pdf.savefig()
            plt.close()

    

        
    

    

                           
