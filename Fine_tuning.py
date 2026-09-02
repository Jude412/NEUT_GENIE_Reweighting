"""The goal of this script is to train any model to reweight the 'original' distribution into the 'target' distribution
with one set of hyperparameters, and to evaluate it. The hyperparameters and metrics are saved in a tensorboard log
file as well as in a csv format, and are gathered afterwards by 'Gather_fine_tuning.py'.

The metrics are computed for every set of analysis parameters given as '--sample_dir NAME path', so that adding a
set to the config file is enough to fine-tune on it."""

#imports
from Metrics_ndim import compute_swd, compute_p_value, chi2_dof, chi2_hist_axis
from Param_sets import ALL_PARAMS_SET, REWEIGHTING_SET, named_paths_from_args
from Sample_io import load_samples
from Train_predict import best_iteration_of, hyperparameters_of, predict_model, train_model
import numpy as np
import pandas as pd
import os
import json
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a reweighting model with one set of hyperparameters and evaluate it.")
    argparser.add_argument('--sample_dir', type=str, required=True,
                           help='Directory where the training original and target samples are stored in parquet format.')
    argparser.add_argument('--swd_distribution', action='append', nargs=2, default=[], metavar=('NAME', 'PATH'),
                           help='Name of a set of analysis parameters and the npy file holding the bootstrapped SWD '
                                'distribution its p-values are computed with. Can be given once per set.')
    argparser.add_argument('--model', type=str, required=True, help='The reweighting model to train.')
    argparser.add_argument('--topology', type=str, required=True, help='The topology to filter the training samples on.')
    argparser.add_argument('--hyperparameters', type=str, required=True, help="Path to the JSON file containing the hyperparameters to train the model with.")
    argparser.add_argument('--logdir', type=str, required=True, help="The directory where the tensorboard log file will be saved.")
    argparser.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    argparser.add_argument("--param_set_dict", type=str, required=True, help="JSON-encoded dict mapping each parameter set name to its list of parameters.")
    argparser.add_argument("--binning_file", type=str, required=True, help="Path to the json file containing the binning information for each parameter.")
    argparser.add_argument("--output_file", type=str, required=True, help="The path to the csv file where the hyperparameters and metrics will be saved in addition to the tensorboard log file.")
    args = argparser.parse_args()

    swd_distributions = named_paths_from_args(args.swd_distribution)
    param_set_dict = json.loads(args.param_set_dict)

    # Load the training and validation samples for the reweighting parameter set
    training_samples = load_samples(args.sample_dir, topology=args.topology, params=param_set_dict[REWEIGHTING_SET], sample_names=("original_train", "target_train", "original_val", "target_val"))

    # Load the testing samples for every parameter set, to compute metrics on them
    testing_samples = {}
    for set_name, param_set in param_set_dict.items():
        testing_samples[set_name] = load_samples(args.sample_dir, topology=args.topology, params=param_set, sample_names=("original_test", "target_test"))

    # Train the model
    hyperparams = hyperparameters_of(args.model, args.hyperparameters)
    model = train_model(args.model, training_samples, hyperparams)
    weights_test = predict_model(args.model, model, testing_samples[REWEIGHTING_SET]["original_test"])

    run_name = "_".join(str(value) for value in hyperparams.values())
    writer = SummaryWriter(os.path.join(args.logdir, f"run_{run_name}_{int(time.time())}"))
    # The trained weights take the pre-weighted original distribution to the pre-weighted target one,
    # so they are multiplied by the pre-weights of the events to give their absolute weights.
    weight_dict = {args.model: weights_test*testing_samples[REWEIGHTING_SET]["original_test"][1]}

    metrics = {}
    if (args.model == 'XGB') | (args.model == 'unnormXGB'):
        best_iter = best_iteration_of(model)
        metrics.update({
            "val_logloss": model.evals_result()['validation_1']['logloss'][best_iter],
            "val_auc": model.evals_result()['validation_1']['auc'][best_iter],
            "train_logloss": model.evals_result()['validation_0']['logloss'][best_iter],
            "train_auc": model.evals_result()['validation_0']['auc'][best_iter],
        })

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    for set_name, samples in testing_samples.items():
        original, _ = samples["original_test"]
        target, target_weight = samples["target_test"]

        if set_name in swd_distributions:
            swd_of_set = compute_swd(original, target, weight_dict,
                                     target_weights=target_weight, n_directions=args.n_directions)
            metrics[f"SWD_{set_name}"] = swd_of_set[args.model]
            metrics[f"p_value_{set_name}"] = compute_p_value(swd_of_set,
                                                             np.load(swd_distributions[set_name]))[args.model]
        else:
            print(f"Warning: no bootstrapped SWD distribution given for the parameter set '{set_name}': "
                  "its SWD and p-value are not computed.")

        metrics[f"chi2_dof_{set_name}"] = chi2_dof(original, target, weight_dict,
                                                   binning_dict=binning_dict,
                                                   target_weights=target_weight,
                                                   list_param_interest=param_set_dict[set_name])[args.model]

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
                                   weight_dict[args.model], axis_number=param_index,
                                   target_weights=all_params_target_weight,
                                   n_bins=n_bins, x_min=x_min, x_max=x_max)
        metrics[f"chi2_dof_{param}"] = chi2/dof if dof > 0 else 0

    for set_name in testing_samples:
        if f"p_value_{set_name}" in metrics:
            print(f"p_value_{set_name} : {metrics[f'p_value_{set_name}']}")

    writer.add_hparams(hyperparams, metrics)

    hparams_metrics_merged = {**hyperparams, **metrics}
    df_row = pd.DataFrame([hparams_metrics_merged])
    df_row.to_csv(args.output_file, index=False)

    print(f"Run with hyperparameters : {hyperparams} is done.")
