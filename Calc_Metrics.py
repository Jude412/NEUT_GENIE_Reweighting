"""The goal of this script is to evaluate the performance of the different reweighting methods.

It takes the test samples of every set of analysis parameters, the weights predicted for them by every
reweighting method, and writes a json file holding the Chi2 statistics, the SWD values and the associated
p-values of every method in every set. The plots displaying those distributions are made by 'Make_plots.py'.

A parameter set is given as '--sample_dir NAME path', and as many sets as wanted can be given: the metrics
of each of them are written under a key named after the set."""

#imports
import os

from Metrics_ndim import chi2_hist_axis, chi2_dof, compute_swd, compute_p_value, chi2_p_value
from Param_sets import ALL_PARAMS_SET, named_paths_from_args
from Sample_io import load_sample, sample_columns
import numpy as np
import argparse
import json

def binning_of(param, binning_dict, default_n_bins=30):
    """Return the (x_min, x_max, n_bins) binning of a parameter, defaulting to an automatic one."""
    if param in binning_dict:
        return binning_dict[param]["x_min"], binning_dict[param]["x_max"], binning_dict[param]["n_bins"]
    return None, None, default_n_bins

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Evaluate the performance of the reweighting methods.')
    argparser.add_argument('--sample_dir', action='append', nargs=2, required=True, metavar=('NAME', 'PATH'),
                           help='Name of a set of analysis parameters and the directory holding its samples. '
                           f"Can be given once per set, and must be given for the '{ALL_PARAMS_SET}' set.")
    argparser.add_argument('--swd_distribution', action='append', nargs=2, default=[], metavar=('NAME', 'PATH'),
                           help='Name of a set of analysis parameters and the npy file holding the bootstrapped SWD '
                           'distribution its p-values are computed with. Can be given once per set.')
    argparser.add_argument('--weights_paths', type=str, required=False,
                           help="Path to the json file holding, for each method, the paths to its predicted weights and to its model.",
                           default='/vols/dune/jmm224/t2knova/reweighting/make_metrics.json')
    argparser.add_argument('--output_file', type=str, required=False, help='Json file to save the metrics in.',
                           default='/vols/dune/jmm224/t2knova/reweighting/saved_metrics/metrics.json')
    argparser.add_argument('--compute_chi2', action=argparse.BooleanOptionalAction, help='Whether to compute the Chi2 statistic or not.')
    argparser.add_argument('--compute_swd', action=argparse.BooleanOptionalAction, help='Whether to compute the SWD metric or not.')
    argparser.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    argparser.add_argument("--binning_file", type=str, required=False, help="Path to the json file containing the binning information for each parameter.",
                           default="/vols/dune/jmm224/t2knova/reweighting/binnings.json")
    args = argparser.parse_args()

    sample_dirs = named_paths_from_args(args.sample_dir)
    swd_distributions = named_paths_from_args(args.swd_distribution)

    if ALL_PARAMS_SET not in sample_dirs:
        raise ValueError(f"No directory given for the '{ALL_PARAMS_SET}' set of every analysis parameter, "
            f"which the per-parameter metrics are computed from. The sets given are {list(sample_dirs)}.")

    # We load the test samples of every parameter set. They all hold the same events, only their
    # parameters differ, so the weights predicted by a method apply to all of them.
    test_samples = {}
    for set_name, sample_dir in sample_dirs.items():
        original_file = os.path.join(sample_dir, "original_test.csv")
        original_test, original_test_weight = load_sample(original_file)
        target_test, target_test_weight = load_sample(os.path.join(sample_dir, "target_test.csv"))
        test_samples[set_name] = {
            "original": original_test,
            "original_weight": original_test_weight,
            "target": target_test,
            "target_weight": target_test_weight,
            "params": sample_columns(original_file),
        }

    all_params_sample = test_samples[ALL_PARAMS_SET]

    with open(args.weights_paths, 'r') as f:
        saved_paths = json.load(f)

    # The trained weights take the pre-weighted original distribution to the pre-weighted target one,
    # so they are multiplied by the pre-weights of the events to give their absolute weights.
    weights_dict = {}
    for method, paths in saved_paths.items():
        weights_dict[method] = np.loadtxt(paths["weights"], delimiter=",")*all_params_sample["original_weight"]

    weights_dict['Original'] = all_params_sample["original_weight"]

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    metrics_dict = {}

    if args.compute_chi2:
        # Chi2 of every analysis parameter taken on its own.
        chi2_1D = {}
        for method in weights_dict:
            chi2_of_params = {}
            for param_index, param in enumerate(all_params_sample["params"]):
                x_min, x_max, n_bins = binning_of(param, binning_dict)
                chi2_val, dof = chi2_hist_axis(all_params_sample["original"], all_params_sample["target"],
                                               weights_dict[method], param_index,
                                               target_weights=all_params_sample["target_weight"],
                                               n_bins=n_bins, x_min=x_min, x_max=x_max)
                chi2_of_params[param] = chi2_val/dof if dof > 0 else 0
                chi2_of_params[param+"_p_value"] = chi2_p_value(chi2_val, dof)
            chi2_1D[method] = chi2_of_params
        metrics_dict['chi2_1D'] = chi2_1D

        # Chi2 of every set of parameters.
        for set_name, sample in test_samples.items():
            chi2_and_dof_of_set = chi2_dof(sample["original"], sample["target"], weights_dict,
                                           target_weights=sample["target_weight"],
                                           binning_dict=binning_dict, list_param_interest=sample["params"],
                                           return_chi2_and_dof=True)
            chi2_of_set = {method: chi2 / dof if dof > 0 else 0
                for method, (chi2, dof) in chi2_and_dof_of_set.items()}
            metrics_dict[f'chi2_{set_name}'] = chi2_of_set
            metrics_dict[f'chi2_{set_name}_p_value'] = {method: chi2_p_value(chi2, dof)
                for method, (chi2, dof) in chi2_and_dof_of_set.items()}

    if args.compute_swd:
        for set_name, sample in test_samples.items():
            if set_name not in swd_distributions:
                print(f"Warning: no bootstrapped SWD distribution given for the parameter set '{set_name}': "
                    "its SWD p-values are not computed.")
                continue

            swd_of_set = compute_swd(sample["original"], sample["target"], weights_dict,
                                     target_weights=sample["target_weight"], n_directions=args.n_directions)
            metrics_dict[f'swd_{set_name}'] = swd_of_set
            metrics_dict[f'p_value_{set_name}'] = compute_p_value(swd_of_set, np.load(swd_distributions[set_name]))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(metrics_dict, f, indent=0)

    print(f"Metrics of the parameter sets {list(test_samples)} saved in {args.output_file}.")
