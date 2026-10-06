"""The goal of this script is to be able to parallelize the bootstrap procedure for the SWD metric. 
The main function is `bootstrap_swd` that we will use only once, and run the script 200 times to get 200 bootstrap samples.
It returns a list saved in a .npy file.

Each bootstrapped distribution holds as many events as the real downstream analysis will use for the
topology (see 'Constants.data_n_events_of'), so that the list is the null distribution of the SWD
between two data-sized samples, which the data-scale p-values of the metric scripts are read against."""

import numpy as np
from Constants import data_n_events_of
from Metrics_ndim import bootstrap_swd, cap_data_n_events
from Sample_io import load_sample
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap SWD metric")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file")
    parser.add_argument("--output_file", type=str, required=True, help="Name of the output file (should end with .npy)")
    parser.add_argument("--distribution", type=str, default="saved_samples/default/target_test.parquet",
                       help="Path to the distribution to bootstrap")
    parser.add_argument('--n_bootstrap', type=int, default=1, help="Number of bootstrap samples to draw")
    parser.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility")
    parser.add_argument('--topology', type=str, required=True, help="Topology to filter the distribution")
    parser.add_argument('--param_set', nargs='+', default=None, help="List of parameters to filter the distribution (optional)")
    parser.add_argument("--data_n_events", type=float, required=True,
                        help="Total number of raw events the real downstream analysis will use for this sample, "
                             "split across topologies by '--target_topology_rates_file'. Every bootstrapped "
                             "distribution holds that many events.")
    parser.add_argument("--target_topology_rates_file", type=str, required=True,
                        help="Path to the json file mapping every topology of this sample to the summed "
                             "PreWeight of its target events, as written by 'Init.py'.")
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    distribution = load_sample(args.distribution, topology=args.topology, params=args.param_set)
    # Capped at the size of the target test sample, as the metric scripts cap their subsamples, so
    # that a topology too small for its data scale is bootstrapped at the size it is evaluated at.
    sample_size = cap_data_n_events(data_n_events_of(args.data_n_events, args.target_topology_rates_file, args.topology),
                                    len(distribution[1]))
    swd_bootstrap = bootstrap_swd(distribution, n_bootstrap=args.n_bootstrap, n_directions=args.n_directions,
                                  sample_size=sample_size)
    np.save(args.output_file, swd_bootstrap)
