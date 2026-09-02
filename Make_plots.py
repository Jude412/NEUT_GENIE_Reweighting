"""Make the plots comparing the original, reweighted and target distributions.

The 1D histograms are made for every analysis parameter, and the 2D ones for the parameters the models
are trained on. The training history of the models that carry one (the XGBoost ones) is plotted from the
model itself, whose path is read from the json file written by 'Training.py'. The metrics displayed on
the plots are computed by 'Metrics_ndim.py', and the ones written to disk by 'Calc_Metrics.py'."""

#imports
import os

from Plots_ndim import labels_of, plot_2D_histogram, plot_histograms, plot_training_history
from Sample_io import load_sample, sample_params
from Train_predict import load_model
import numpy as np
import argparse
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Plot the reweighted distributions of the different reweighting methods.')
    argparser.add_argument('--sample_dir', type=str, required=True,
                           help='Directory holding the samples the 1D histograms are made from (usually the samples of every analysis parameter).')
    argparser.add_argument('--topology', type=str, required=True, help='The topology to filter the samples on.')
    argparser.add_argument('--weights_paths', type=str, required=True,
                           help="Path to the json file holding, for each method, the paths to its predicted weights and to its model.")
    argparser.add_argument('--output_dir', type=str, required=True, help='Directory to save the plots in.')
    argparser.add_argument('--make_1D_plots', action=argparse.BooleanOptionalAction, help='Whether to make the 1D plots or not.')
    argparser.add_argument('--make_2D_plots', action=argparse.BooleanOptionalAction, help='Whether to make the 2D plots or not.')
    argparser.add_argument('--make_training_history', action=argparse.BooleanOptionalAction, default=True,
                           help='Whether to plot the training history of the models carrying one or not.')
    argparser.add_argument("--binning_file", type=str, required=True,
                           help="Path to the json file containing the binning information for each parameter.")
    args = argparser.parse_args()

    with open(args.weights_paths, 'r') as f:
        saved_paths = json.load(f)

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    def weights_of(original_weight):
        """Absolute weights of the events of a sample, for every method and for the original distribution.

        The trained weights take the pre-weighted original distribution to the pre-weighted target one,
        so they are multiplied by the pre-weights of the events."""
        weights_dict = {method: np.load(paths["weights"])*original_weight
                        for method, paths in saved_paths.items()}
        weights_dict['Original'] = original_weight
        return weights_dict

    original_file = os.path.join(args.sample_dir, "original_test.parquet")
    target_file = os.path.join(args.sample_dir, "target_test.parquet")
    original_test, original_test_weight = load_sample(original_file, topology=args.topology)
    target_test, target_test_weight = load_sample(target_file, topology=args.topology)
    params = sample_params(original_file)

    if args.make_1D_plots:
        plot_histograms(original_test, target_test, weights_of(original_test_weight),
                        dict_binning=binning_dict,
                        original_weights=original_test_weight,
                        target_weights=target_test_weight,
                        xlabels=labels_of(params),
                        variables=params,
                        output_file=os.path.join(args.output_dir, "1Dhist.pdf"))

    if args.make_2D_plots:
        plot_2D_histogram(original_test, target_test, weights_of(original_test_weight),
                          target_weights=target_test_weight,
                          xlabels=labels_of(params),
                          nbins=30,
                          pull=True,
                          output_file=os.path.join(args.output_dir, "2Dhist.pdf"))

    if args.make_training_history:
        for method, paths in saved_paths.items():
            model = load_model(method, paths["model"])
            plot_training_history(model, os.path.join(args.output_dir, f"{method}_training_history.pdf"),
                                  title=f"Training history of {method}")

    print(f"Plots saved in {args.output_dir}.")
