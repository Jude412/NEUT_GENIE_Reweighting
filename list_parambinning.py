import json
import os
import argparse

args = argparse.ArgumentParser(description='Generate json files for each binning hyperparameter combination.')
args.add_argument('--output_dir', type=str, default='hps', help='Directory to save the json files.')
args = args.parse_args()

n_bins = [5, 6, 7, 8, 9, 10, 11, 12]
n_neighs = [0, 1, 2]

sub_file_template = ""

counter = 0

os.makedirs(os.path.join(args.output_dir, "binning"), exist_ok=True)

for n_bin in n_bins:
    for n_neigh in n_neighs:
        with open(os.path.join(args.output_dir, f"binning/binning_hp_{counter}.json"), "w") as f:
            json.dump({"n_bins": n_bin, "n_neighs": n_neigh}, f)    
        sub_file_template += os.path.join("/vols/dune/jmm224/t2knova/reweighting/", args.output_dir, f"binning/binning_hp_{counter}.json") + "\n"
        counter += 1

sub_file_template += ")"

with open(os.path.join(args.output_dir, "binning/binning_job.sub"), "w") as f:
    f.write(sub_file_template)
