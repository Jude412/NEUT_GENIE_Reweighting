"""Make the plots comparing the original, reweighted and target distributions.

The 1D histograms are made for every analysis parameter, and the 2D ones for the parameters the models
are trained on. Every model is loaded from the symlink 'Gather_metrics.py' writes for its best
hyperparameter set, and its weights are predicted here rather than read from disk. The training history
of the models that carry one (the XGBoost ones) is plotted from the model itself. The metrics displayed
on the plots are computed by 'Metrics_ndim.py', and the ones written to disk by 'Compute_metrics.py'."""

#imports
import os

from Plots_ndim import labels_of, plot_2D_histogram, plot_histograms, plot_training_history
from Sample_io import load_sample, load_samples, sample_params
from Train_predict import load_model, predict_model
import argparse
import json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Plot the reweighted distributions of the different reweighting methods.')
    argparser.add_argument('--sample_dir', type=str, required=True,
                           help='Directory holding the samples the 1D histograms are made from (usually the samples of every analysis parameter).')
    argparser.add_argument('--topology', type=str, required=True, help='The topology to filter the samples on.')
    argparser.add_argument('--model_dir', type=str, required=True,
                           help="Directory holding the best model of every reweighting method, as written by 'Gather_metrics.py'.")
    argparser.add_argument('--model_list', required=True, nargs='+', help='List of the reweighting models to plot.')
    argparser.add_argument('--reweight_params', required=True, nargs='+',
                           help='List of analysis parameters the models were trained on, to predict their weights with.')
    argparser.add_argument('--output_dir', type=str, required=True, help='Directory to save the plots in.')
    argparser.add_argument('--make_1D_plots', action=argparse.BooleanOptionalAction, help='Whether to make the 1D plots or not.')
    argparser.add_argument('--make_2D_plots', action=argparse.BooleanOptionalAction, help='Whether to make the 2D plots or not.')
    argparser.add_argument('--make_training_history', action=argparse.BooleanOptionalAction, default=True,
                           help='Whether to plot the training history of the models carrying one or not.')
    argparser.add_argument("--binning_file", type=str, required=True,
                           help="Path to the json file containing the binning information for each parameter.")
    args = argparser.parse_args()

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    models = {model: load_model(model, os.path.join(args.model_dir, model, model)) for model in args.model_list}

    # The models were trained on the reweighting parameters only, so their weights are predicted from
    # that subset of the samples, whichever parameters the plots below are made on: every parameter set
    # shares the same events, so the weights apply unchanged regardless of which columns are read.
    reweight_samples = load_samples(args.sample_dir, topology=args.topology, params=args.reweight_params,
                                    sample_names=("original_test",))

    def weights_of(original_weight):
        """Absolute weights of the events of a sample, for every method and for the original distribution.

        The trained weights take the pre-weighted original distribution to the pre-weighted target one,
        so they are multiplied by the pre-weights of the events."""
        weights_dict = {model: predict_model(model, trained_model, reweight_samples["original_test"])*original_weight
                        for model, trained_model in models.items()}
        weights_dict['Original'] = original_weight
        return weights_dict

    original_file = os.path.join(args.sample_dir, "original_test.parquet")
    target_file = os.path.join(args.sample_dir, "target_test.parquet")
    original_test, original_test_weight = load_sample(original_file, topology=args.topology)
    target_test, target_test_weight = load_sample(target_file, topology=args.topology)
    params = sample_params(original_file)
    print(f"Plotting {len(params)} params: {params}")
    print(f"Original shape = {original_test.shape}, Target shape = {target_test.shape}")

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
        for model, trained_model in models.items():
            plot_training_history(trained_model, os.path.join(args.output_dir, f"{model}_training_history.pdf"),
                                  title=f"Training history of {model}")

    print(f"Plots saved in {args.output_dir}.")
