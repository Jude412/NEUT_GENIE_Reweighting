"""The goal of this script is to calculate the un-reweighted ('Original') baseline metrics of a
sample and topology: the same quantities 'Compute_metrics.py' computes for a trained model, taken
without any reweighting, so that a model's metrics can be read against how far apart the two
distributions started.

They depend only on the test samples and the binning, never on a model or its hyperparameters, so
they are computed once here rather than in every one of the thousands of 'compute_metrics' jobs of
a hyperparameter grid. 'Compute_metrics.py' reads the file written here and merges its columns
into its own, which is what keeps every run reporting one metric schema."""

#imports
from Constants import data_n_events_of
from Metrics_ndim import cap_data_n_events, compute_all_metrics, data_subsample_indices
from Param_sets import REWEIGHTING_SET, named_paths_from_args
from Sample_io import load_samples
import pandas as pd
import json
import argparse

# The prefix the baseline columns carry, so that they do not collide with the model's own metrics
# once they are merged into a run's metrics file. 'Plot_hyperparameters.py' reads them back by it.
BASELINE_PREFIX = "Original_"
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
                           help="Seed the data-sized subsamples are drawn from. It must be the one the 'compute_metrics' jobs "
                                "of this sample and topology are given, so that they are all evaluated on the same events.")
    argparser.add_argument("--output_file", type=str, required=True, help="The path to the csv file where the baseline metrics will be saved.")
    args = argparser.parse_args()

    swd_distributions = named_paths_from_args(args.swd_distribution)
    param_set_dict = json.loads(args.param_set_dict)

    # Load the testing samples for every parameter set, to compute metrics on them
    testing_samples = {}
    for set_name, param_set in param_set_dict.items():
        testing_samples[set_name] = load_samples(args.sample_dir, topology=args.topology, params=param_set, sample_names=("original_test", "target_test"))

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

    # The baseline is the original distribution carrying nothing but its own pre-weights, i.e. the
    # weights a model's predictions are multiplied by to give the absolute weights of its events.
    metrics = compute_all_metrics(testing_samples, testing_samples[REWEIGHTING_SET]["original_test"][1],
                                  param_set_dict, binning_dict, swd_distributions,
                                  swd_seed=args.swd_seed, n_directions=args.n_directions,
                                  n_events=n_events, subsamples=subsamples)
    metrics = {f"{BASELINE_PREFIX}{name}": value for name, value in metrics.items()}

    for set_name in testing_samples:
        print(f"{BASELINE_PREFIX}data_p_value_{set_name} : {metrics[f'{BASELINE_PREFIX}data_p_value_{set_name}']}")

    pd.DataFrame([metrics]).to_csv(args.output_file, index=False)

    print(f"The baseline metrics of the topology {args.topology} are done.")
