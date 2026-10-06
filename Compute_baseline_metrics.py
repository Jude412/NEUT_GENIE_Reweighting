"""The goal of this script is to calculate the un-reweighted ('Original') baseline metrics of a
sample and topology: the same quantities 'Compute_metrics.py' computes for a trained model, taken
without any reweighting, so that a model's metrics can be read against how far apart the two
distributions started.

They depend only on the test samples and the binning, never on a model or its hyperparameters, so
they are computed once here rather than in every one of the thousands of 'compute_metrics' jobs of
a hyperparameter grid. 'Compute_metrics.py' reads the file written here and merges its columns
into its own, which is what keeps every run reporting one metric schema."""

#imports
from Metrics_ndim import (compute_swd, compute_p_value, chi2_dof, chi2_hist_axis, chi2_p_value,
                          swd_direction_rng)
from Param_sets import ALL_PARAMS_SET, REWEIGHTING_SET, named_paths_from_args
from Sample_io import load_samples
import numpy as np
import pandas as pd
import json
import argparse

# The prefix the baseline columns carry, so that they do not collide with the model's own metrics
# once they are merged into a run's metrics file. 'Plot_hyperparameters.py' reads them back by it.
BASELINE_PREFIX = "Original_"
# The key the un-reweighted weights are held under in the dictionaries the metric functions take.
BASELINE_KEY = "Original"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Compute the un-reweighted baseline metrics of a sample and topology.")
    argparser.add_argument('--sample_dir', type=str, required=True,
                           help='Directory where the training original and target samples are stored in parquet format.')
    argparser.add_argument('--swd_distribution', action='append', nargs=2, default=[], metavar=('NAME', 'PATH'),
                           help='Name of a set of analysis parameters and the npy file holding the bootstrapped SWD '
                                'distribution its p-values are computed with. Can be given once per set.')
    argparser.add_argument('--topology', type=str, required=True, help='The topology to filter the training samples on.')
    argparser.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    argparser.add_argument("--param_set_dict", type=str, required=True, help="JSON-encoded dict mapping each parameter set name to its list of parameters.")
    argparser.add_argument("--binning_file", type=str, required=True, help="Path to the json file containing the binning information for each parameter.")
    argparser.add_argument("--swd_seed", type=int, required=True,
                           help="Seed the SWD projection directions are drawn from. It must be the one the "
                                "'compute_metrics' jobs of this sample and topology are given, so that their "
                                "distances and these are measured along the same directions.")
    argparser.add_argument("--output_file", type=str, required=True, help="The path to the csv file where the baseline metrics will be saved.")
    args = argparser.parse_args()

    swd_distributions = named_paths_from_args(args.swd_distribution)
    param_set_dict = json.loads(args.param_set_dict)

    # Load the testing samples for every parameter set, to compute metrics on them
    testing_samples = {}
    for set_name, param_set in param_set_dict.items():
        testing_samples[set_name] = load_samples(args.sample_dir, topology=args.topology, params=param_set, sample_names=("original_test", "target_test"))

    # The baseline is the original distribution carrying nothing but its own pre-weights, i.e. the
    # weights a model's predictions are multiplied by to give the absolute weights of its events.
    # Every parameter set is read from the same files with the same topology filter, so the sets
    # differ in their columns and not in their rows: one weight array applies to all of them.
    weight_dict = {BASELINE_KEY: testing_samples[REWEIGHTING_SET]["original_test"][1]}

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    # Normalisation of the un-reweighted original relative to the target, on the test sample.
    metrics = {f"{BASELINE_PREFIX}norm_test": np.sum(weight_dict[BASELINE_KEY]) / np.sum(testing_samples[REWEIGHTING_SET]["target_test"][1])}
    for set_name, samples in testing_samples.items():
        original, _ = samples["original_test"]
        target, target_weight = samples["target_test"]

        metrics[f"{BASELINE_PREFIX}SWD_{set_name}"] = float("nan")
        metrics[f"{BASELINE_PREFIX}p_value_{set_name}"] = float("nan")
        if set_name in swd_distributions:
            swd_of_set = compute_swd(original, target, weight_dict,
                                     target_weights=target_weight, n_directions=args.n_directions,
                                     rng=swd_direction_rng(args.swd_seed, set_name))
            p_value_of_set = compute_p_value(swd_of_set, np.load(swd_distributions[set_name]))
            metrics[f"{BASELINE_PREFIX}SWD_{set_name}"] = swd_of_set[BASELINE_KEY]
            metrics[f"{BASELINE_PREFIX}p_value_{set_name}"] = p_value_of_set[BASELINE_KEY]
        else:
            print(f"Warning: no bootstrapped SWD distribution given for the parameter set '{set_name}': "
                  "its SWD and p-value are not computed.")

        chi2_and_dof_of_set = chi2_dof(original, target, weight_dict,
                                       binning_dict=binning_dict,
                                       target_weights=target_weight,
                                       list_param_interest=param_set_dict[set_name],
                                       return_chi2_and_dof=True)
        chi2, dof = chi2_and_dof_of_set[BASELINE_KEY]
        metrics[f"{BASELINE_PREFIX}chi2_dof_{set_name}"] = chi2/dof if dof > 0 else 0
        metrics[f"{BASELINE_PREFIX}chi2_p_value_{set_name}"] = chi2_p_value(chi2, dof)

    # Chi2 of every analysis parameter taken on its own.
    all_params_original, _ = testing_samples[ALL_PARAMS_SET]["original_test"]
    all_params_target, all_params_target_weight = testing_samples[ALL_PARAMS_SET]["target_test"]
    for param_index, param in enumerate(param_set_dict[ALL_PARAMS_SET]):
        if param in binning_dict:
            x_min = binning_dict[param]["x_min"]
            x_max = binning_dict[param]["x_max"]
            n_bins = binning_dict[param]["n_bins"]
        else:
            x_min = None
            x_max = None
            n_bins = 30
        chi2, dof = chi2_hist_axis(all_params_original, all_params_target,
                                   weight_dict[BASELINE_KEY], axis_number=param_index,
                                   target_weights=all_params_target_weight,
                                   n_bins=n_bins, x_min=x_min, x_max=x_max)
        metrics[f"{BASELINE_PREFIX}chi2_dof_{param}"] = chi2/dof if dof > 0 else 0
        metrics[f"{BASELINE_PREFIX}chi2_p_value_{param}"] = chi2_p_value(chi2, dof)

    for set_name in testing_samples:
        print(f"{BASELINE_PREFIX}p_value_{set_name} : {metrics[f'{BASELINE_PREFIX}p_value_{set_name}']}")

    pd.DataFrame([metrics]).to_csv(args.output_file, index=False)

    print(f"The baseline metrics of the topology {args.topology} are done.")
