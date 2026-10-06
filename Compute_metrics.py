"""The goal of this script is to calculate the metrics of a reweighting model trained on a single set of hyperparameters.

The metrics of the un-reweighted ('Original') baseline are reported alongside them, so that a model's metrics can be
read against how far apart the two distributions started, but they are not computed here: they depend on neither the
model nor its hyperparameters, so 'Compute_baseline_metrics.py' computes them once for the whole hyperparameter grid
of a sample and topology and this script merges that file's columns into its own."""

#imports
from Constants import data_n_events_of
from Metrics_ndim import cap_data_n_events, compute_all_metrics, data_subsample_indices
from Param_sets import REWEIGHTING_SET, named_paths_from_args
from Sample_io import load_samples
from Train_predict import best_iteration_of, predict_model, train_model, load_model
import pandas as pd
import os
import json
from torch.utils.tensorboard import SummaryWriter
import argparse

# The prefix the un-reweighted baseline of a metric carries, and the metrics that have no baseline
# because they are read off the training history of the model rather than computed from a sample.
BASELINE_PREFIX = "Original_"
TRAINING_METRICS = ("val_logloss", "val_auc", "train_logloss", "train_auc")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a reweighting model with one set of hyperparameters and evaluate it.")
    argparser.add_argument('--sample_dir', type=str, required=True,
                           help='Directory where the training original and target samples are stored in parquet format.')
    argparser.add_argument('--swd_distribution', action='append', nargs=2, default=[], metavar=('NAME', 'PATH'),
                           help='Name of a set of analysis parameters and the npy file holding the bootstrapped SWD '
                                'distribution its p-values are computed with. Can be given once per set.')
    argparser.add_argument('--model_dir', type=str, required=True, help='Directory where the trained model is stored.')
    argparser.add_argument('--model', type=str, required=True, help='The reweighting model to train.')
    argparser.add_argument('--topology', type=str, required=True, help='The topology to filter the training samples on.')
    argparser.add_argument('--logdir', type=str, required=True,
                           help="The directory the tensorboard logs of this run are written to. It "
                                "holds this run alone, so it is expected to be empty: the workflow "
                                "declares it as an output and so clears it before every rerun.")
    argparser.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    argparser.add_argument("--param_set_dict", type=str, required=True, help="JSON-encoded dict mapping each parameter set name to its list of parameters.")
    argparser.add_argument("--binning_file", type=str, required=True, help="Path to the json file containing the binning information for each parameter.")
    argparser.add_argument("--baseline_metrics", type=str, required=True,
                           help="Path to the csv file holding the un-reweighted baseline metrics of this "
                                "sample and topology, written by 'Compute_baseline_metrics.py'. Its columns "
                                "are merged into the ones computed here.")
    argparser.add_argument("--swd_seed", type=int, required=True,
                           help="Seed the SWD projection directions are drawn from. It must be the one the "
                                "baseline of this sample and topology was computed with, so that the two sets "
                                "of distances are measured along the same directions.")
    argparser.add_argument("--data_n_events", type=float, required=True,
                           help="Total number of raw events the real downstream analysis will use for this "
                                "sample, split across topologies by '--target_topology_rates_file'. The "
                                "data-scale metrics are computed at that many events.")
    argparser.add_argument("--target_topology_rates_file", type=str, required=True,
                           help="Path to the json file mapping every topology of this sample to the summed "
                                "PreWeight of its target events, as written by 'Init.py'.")
    argparser.add_argument("--n_subsamples", type=int, required=True,
                           help="Number of data-sized subsamples the data-scale SWD and p-value are averaged over.")
    argparser.add_argument("--subsample_seed", type=int, required=True,
                           help="Seed the data-sized subsamples are drawn from. It must be the one the baseline "
                                "of this sample and topology was computed with, so that they are all evaluated on the same events.")
    argparser.add_argument("--output_file", type=str, required=True, help="The path to the csv file where the hyperparameters and metrics will be saved in addition to the tensorboard log file.")
    args = argparser.parse_args()

    swd_distributions = named_paths_from_args(args.swd_distribution)
    param_set_dict = json.loads(args.param_set_dict)

    # Read the hyperparameters the model was actually trained with (a '..._frac' hyperparameter of the
    # grid file resolved to its absolute value, see 'Train_predict.resolve_fractional_hyperparameters'),
    # written by 'Train_model.py' alongside the model itself.
    with open(os.path.join(args.model_dir, f"{args.model}_resolved_hyperparameters.json")) as f:
        hyperparams = json.load(f)

    # Every run reports the same set of metric keys, whatever the model type or the sets a
    # bootstrapped SWD distribution was given for, so that TensorBoard's HParams table (and the
    # aggregated metrics.csv rows) can compare runs on a consistent schema instead of silently
    # missing columns. The 'Original_' half of that schema is not computed here: it does not
    # depend on the model, so it is computed once per sample and topology by
    # 'Compute_baseline_metrics.py' and merged in below, before anything is written out.
    metrics = {name: float("nan") for name in TRAINING_METRICS}

    # The model is loaded, used and dropped before the samples of the other parameter sets are
    # read in, rather than after all of them are already resident. A deep XGBoost model of a large
    # grid takes a couple of gigabytes to parse back out of its json file, which is enough to push
    # a job over the 4 GB the grid allows it when it is held alongside every sample at once: run 4
    # lost eleven hyperparameter sets that way, all of them the largest models of the grid. Holding
    # them one after the other instead leaves the peak at whichever of the two is larger, which is
    # the samples, whatever the size of the model.
    testing_samples = {REWEIGHTING_SET: load_samples(args.sample_dir, topology=args.topology,
                                                     params=param_set_dict[REWEIGHTING_SET],
                                                     sample_names=("original_test", "target_test"))}

    model_path = os.path.join(args.model_dir, f"{args.model}")
    model = load_model(args.model, model_path)
    weights_test = predict_model(args.model, model, testing_samples[REWEIGHTING_SET]["original_test"])
    if (args.model == 'XGB') | (args.model == 'unnormXGB'):
        best_iter = best_iteration_of(model)
        metrics.update({
            "val_logloss": model.evals_result()['validation_1']['logloss'][best_iter],
            "val_auc": model.evals_result()['validation_1']['auc'][best_iter],
            "train_logloss": model.evals_result()['validation_0']['logloss'][best_iter],
            "train_auc": model.evals_result()['validation_0']['auc'][best_iter],
        })
    # Nothing below reads the model again: the weights it gave are what the metrics are computed
    # from. Dropping it here is what keeps it from being resident alongside the samples below.
    del model

    for set_name, param_set in param_set_dict.items():
        if set_name not in testing_samples:
            testing_samples[set_name] = load_samples(args.sample_dir, topology=args.topology, params=param_set, sample_names=("original_test", "target_test"))
    # Put the sets back in the order of the config file, which the reweighting set was taken out
    # of above, so that the metrics are written in the column order they have always been in.
    testing_samples = {set_name: testing_samples[set_name] for set_name in param_set_dict}

    writer = SummaryWriter(args.logdir)

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    # The data-sized subsamples every metric prefixed 'data_' is computed on. Every parameter set is
    # read from the same files with the same topology filter, so they share their events and one set
    # of indices applies to all of them.
    n_original = len(testing_samples[REWEIGHTING_SET]["original_test"][1])
    n_target = len(testing_samples[REWEIGHTING_SET]["target_test"][1])
    n_events = cap_data_n_events(data_n_events_of(args.data_n_events, args.target_topology_rates_file, args.topology),
                                 min(n_original, n_target))
    subsamples = data_subsample_indices(n_original, n_target, n_events, args.n_subsamples, args.subsample_seed)

    # The trained weights take the pre-weighted original distribution to the pre-weighted target one,
    # so they are multiplied by the pre-weights of the events to give their absolute weights.
    metrics.update(compute_all_metrics(testing_samples, weights_test*testing_samples[REWEIGHTING_SET]["original_test"][1],
                                       param_set_dict, binning_dict, swd_distributions,
                                       swd_seed=args.swd_seed, n_directions=args.n_directions,
                                       n_events=n_events, subsamples=subsamples))

    for set_name in testing_samples:
        print(f"data_p_value_{set_name} : {metrics[f'data_p_value_{set_name}']}")

    # The un-reweighted baseline of this sample and topology, merged in before either output is
    # written so that the tensorboard HParams table and the metrics file both carry the model's
    # metrics and the baseline they are read against. Every metric computed from a sample is
    # expected to have one, and a missing column is raised rather than left to become a silent
    # NaN in the gathered grid: the two scripts compute their halves separately and this is what
    # catches them drifting apart.
    baseline = pd.read_csv(args.baseline_metrics).iloc[0].to_dict()
    missing = [f"{BASELINE_PREFIX}{name}" for name in metrics
               if name not in TRAINING_METRICS and f"{BASELINE_PREFIX}{name}" not in baseline]
    if missing:
        raise ValueError(f"The baseline metrics file '{args.baseline_metrics}' is missing the "
                         f"column(s) {missing}. It is written by 'Compute_baseline_metrics.py', "
                         "which must report a baseline for every metric computed here.")
    # Each baseline goes next to the metric it is the baseline of, rather than all of them after
    # all of the metrics, which is the column order the metrics files have always been written in.
    merged = {}
    for name, value in metrics.items():
        merged[name] = value
        if f"{BASELINE_PREFIX}{name}" in baseline:
            merged[f"{BASELINE_PREFIX}{name}"] = baseline[f"{BASELINE_PREFIX}{name}"]
    metrics = merged

    # 'run_name' is set to '.' so that the hparams and the metrics are written directly into the log
    # directory, instead of a separately-timestamped subdirectory of it that would be a run of its own
    # and would leave the log directory itself holding nothing but an empty event file.
    writer.add_hparams(hyperparams, metrics, run_name=".")
    writer.close()

    hparams_metrics_merged = {**hyperparams, **metrics}
    df_row = pd.DataFrame([hparams_metrics_merged])
    df_row.to_csv(args.output_file, index=False)

    print(f"Run with hyperparameters : {hyperparams} is done.")
