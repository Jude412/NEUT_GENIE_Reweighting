import json
import os
import argparse

args = argparse.ArgumentParser(description='Generate json files for each XGB hyperparameter combination.')
args.add_argument('--output_dir', type=str, default='hps', help='Directory to save the json files.')
args = args.parse_args()

############## Original values ###############
n_estimators_list = [100, 150, 200]
max_depth_list = [3, 5, 7, 9]
learning_rate_list = [0.01, 0.05, 0.1, 0.2]
subsample_list = [1]
gamma_list = [0, 0.5] #default 0
min_child_weight_list = [1] #default 1
lambda_list = [1] #default 1
alpha_list = [0] #default 0
early_stopping_rounds_list = [10]
############## Testing values ###############
# n_estimators_list = [100]
# max_depth_list = [3]
# learning_rate_list = [0.05]
# subsample_list = [1]
# gamma_list = [0] #default 0
# min_child_weight_list = [1] #default 1
# lambda_list = [1] #default 1
# alpha_list = [0] #default 0
# early_stopping_rounds_list = [10]
#############################################

sub_file_template = ""

counter = 0

os.makedirs(os.path.join(args.output_dir, "XGB"), exist_ok=True)
for n_estimator in n_estimators_list:
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            for subsample in subsample_list:
                for gamma in gamma_list:
                    for min_child_weight in min_child_weight_list:
                        for lambda_val in lambda_list:
                            for alpha_val in alpha_list:
                                for early_stopping_rounds in early_stopping_rounds_list:
                                    with open(os.path.join(args.output_dir, f"XGB/XGB_hp_{counter}.json"), "w") as f:
                                        json.dump({"n_estimators": n_estimator, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample, "gamma": gamma, "min_child_weight": min_child_weight, "lambda": lambda_val, "alpha": alpha_val, "early_stopping_rounds": early_stopping_rounds}, f)

                                    sub_file_template += os.path.join("/vols/dune/jmm224/t2knova/reweighting/", args.output_dir, f"XGB/XGB_hp_{counter}.json") + "\n"
                                    counter += 1

sub_file_template += ")"

with open(os.path.join(args.output_dir, "XGB/XGB_job.sub"), "w") as f:
    f.write(sub_file_template)
