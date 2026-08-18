import json
import os
import argparse

args = argparse.ArgumentParser(description='Generate json files for each GBR hyperparameter combination.')
args.add_argument('--output_dir', type=str, default='hps', help='Directory to save the json files.')
args = args.parse_args()

n_estimators_list = [80, 100, 120]
max_depth_list = [3, 4, 5, 6, 7]
learning_rate_list = [0.01, 0.05, 0.1]
min_samples_leaf_list = [50, 100]
loss_regularization_list = [2, 4, 6]

sub_file_template = ""

counter = 0

os.makedirs(os.path.join(args.output_dir, "GBR"), exist_ok=True)
for n_estimator in n_estimators_list:
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            for min_samples_leaf in min_samples_leaf_list:
                for loss_regularization in loss_regularization_list:
                    with open(os.path.join(args.output_dir, f"GBR/GBR_hp_{counter}.json"), "w") as f:
                        json.dump({"n_estimators": n_estimator, "learning_rate": learning_rate, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "loss_regularization": loss_regularization}, f)
                    sub_file_template += os.path.join("/vols/dune/jmm224/t2knova/reweighting/", args.output_dir, f"GBR/GBR_hp_{counter}.json") + "\n"
                    counter += 1

sub_file_template += ")"

with open(os.path.join(args.output_dir, "GBR/GBR_job.sub"), "w") as f:
    f.write(sub_file_template)
