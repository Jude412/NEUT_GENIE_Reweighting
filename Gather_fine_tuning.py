import numpy as np
import argparse
import pandas as pd
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate fine-tuning metrics from multiple runs.')
    parser.add_argument('--input_dir', help='Input tensorboard directory containing fine-tuning metrics.')
    parser.add_argument('--output_file', help='Output json file for the aggregated metrics.')
    parser.add_argument('--model_list', nargs='+', help='List of model names corresponding to the input files.')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_file = args.output_file
    model_list = args.model_list

    model_best_set_hyperparameters = {}
    for model in model_list:
        if model == "binning":
            df = pd.read_csv(os.path.join(input_dir, f"{model}/Hyperparameters_metrics.csv"))
            best_set = df.loc[df['SWD_3D'].idxmin()]
            model_best_set_hyperparameters[model] = {
                "n_bins" : int(best_set['n_bins']),
                "n_neighs" : int(best_set['n_neighs'])
            }
        elif model == "XGB":
            df = pd.read_csv(os.path.join(input_dir, f"{model}/Hyperparameters_metrics.csv"))
            best_set = df.loc[df['p_value_8D'].idxmax()]
            model_best_set_hyperparameters[model] = {
                "n_estimators" : int(best_set['n_estimators']),
                 "gamma" : float(best_set['gamma']),
                 "alpha" : float(best_set['alpha']),
                 "lambda" : float(best_set['lambda']),
                 "max_depth" : int(best_set['max_depth']),
                 "learning_rate" : float(best_set['learning_rate']),
                "subsample" : float(best_set['subsample']), 
                "early_stopping_rounds" : int(best_set['early_stopping_rounds'])
            }
        else:
            print(f"Model {model} not recognized. Skipping.")
            continue
    
    with open(output_file, 'w') as f:
        json.dump(model_best_set_hyperparameters, f)

