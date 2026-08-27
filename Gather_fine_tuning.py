"""Pick, for every model, the hyperparameter set giving the best metric of the fine-tuning runs.

The hyperparameters and metrics of every run are written by 'Fine_tuning.py' in
'{model}/run_*_metrics.csv' files. This script reads them, keeps the row giving the best value
of the metric the model is selected on, and writes the hyperparameters of that row as the json
file the final training uses.

The hyperparameter names are read from the grid file rather than hard-coded, so that changing
the grid of a model is enough to change the hyperparameters that are scanned and selected."""

# imports
from List_hyperparameters import grid_of
import argparse
import json
import os
import glob
import pandas as pd

def cast_like(value, reference):
    """Return a json-serialisable value cast to the type of the reference value of the grid.

    The metrics and the hyperparameters travel through a csv file, so the values read back are
    numpy ones (and integers are read as floats): they are cast back to the type they are given
    with in the grid file."""
    if isinstance(reference, bool):
        return bool(value)
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return type(reference)(value)

def best_hyperparameters(metrics_files, grid_point, metric, direction):
    """Return the hyperparameters of the run giving the best value of the given metric.

    'grid_point' is any point of the grid of the model: it gives the names of its
    hyperparameters and the type their values are given with."""
    if not metrics_files:
        raise ValueError("No fine-tuning metrics files found.")

    runs = pd.concat([pd.read_csv(metrics_file) for metrics_file in metrics_files], ignore_index=True)

    if metric not in runs.columns:
        raise ValueError(f"The metric '{metric}' the best hyperparameter set is picked with was not "
                         f"found in {metrics_files}, which holds {list(runs.columns)}.")

    missing = [name for name in grid_point if name not in runs.columns]
    if missing:
        raise ValueError(f"The hyperparameters {missing} of the grid file were not found in "
                         f"{metrics_files}, which holds {list(runs.columns)}.")

    if direction == "min":
        best_run = runs.loc[runs[metric].idxmin()]
    elif direction == "max":
        best_run = runs.loc[runs[metric].idxmax()]
    else:
        raise ValueError(f"Unknown direction '{direction}' for the metric '{metric}': "
                         "please choose from 'min' and 'max'.")

    return {name: cast_like(best_run[name], reference) for name, reference in grid_point.items()}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Gather the fine-tuning runs of every model and keep the best hyperparameter set of each.")
    argparser.add_argument("--input_dir", required=True, help="Tensorboard directory holding the fine-tuning metrics of every model.")
    argparser.add_argument("--output_file", required=True, help="Output json file for the best hyperparameter set of every model.")
    argparser.add_argument("--grid_file", required=True, help="Path to the json file holding the hyperparameter grid of every model.")
    argparser.add_argument("--selection", action="append", nargs=3, required=True, metavar=("MODEL", "METRIC", "DIRECTION"),
                           help="Model, metric its best hyperparameter set is picked with, and whether that metric is "
                                "minimised ('min') or maximised ('max'). Can be given several times, once per model.")
    args = argparser.parse_args()

    with open(args.grid_file) as f:
        grids = json.load(f)

    best_hyperparameters_of_models = {}
    for model, metric, direction in args.selection:
        if model not in grids:
            raise ValueError(f"No hyperparameter grid given for the model '{model}' in {args.grid_file}. "
                             f"The grids given are those of {list(grids)}.")

        metrics_files = sorted(glob.glob(os.path.join(args.input_dir, model, "run_*_metrics.csv")))
        best_hyperparameters_of_models[model] = best_hyperparameters(metrics_files, grid_of(grids[model])[0],
                                                                    metric, direction)
        print(f"Best hyperparameter set of the model {model} ({direction} {metric}): "
              f"{best_hyperparameters_of_models[model]}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(best_hyperparameters_of_models, f, indent=0)
