"""The goal of this script is to be able to parallelize the bootstrap procedure for the SWD metric. 
The main function is `bootstrap_swd` that we will use only once, and run the script 200 times to get 200 bootstrap samples.
It returns a list saved in a .npy file."""

import numpy as np
from Metrics_ndim import bootstrap_swd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap SWD metric")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file")
    parser.add_argument("--output_file", type=str, required=True, help="Name of the output file (should end with .npy)")
    parser.add_argument("--distribution", type=str, default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/8D/target_test.csv",
                       help="Path to the distribution to bootstrap")
    parser.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    distribution = np.loadtxt(args.distribution, delimiter=',')
    swd_bootstrap = bootstrap_swd(distribution, n_bootstrap=1, n_directions=args.n_directions)
    np.save(args.output_file, swd_bootstrap)
