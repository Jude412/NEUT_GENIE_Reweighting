"""Pick, for every model, the hyperparameter set giving the best metric of the fine-tuning runs.

The hyperparameters and metrics of every run are written by 'Compute_metrics.py' in
'{model}/{run_id}/metrics.csv' files. This script reads them, keeps the row giving the best value
of the metric the model is selected on, and writes its hyperparameters and every metric it holds
to the output json file, alongside a symlink to that run's model (named after the model, without
the run_id, so that it is the one path downstream scripts need to know about).

The hyperparameter names are read from the grid file rather than hard-coded, so that changing
the grid of a model is enough to change the hyperparameters that are scanned and selected."""

# imports
from List_hyperparameters import grid_of
from Train_predict import model_extension
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
    """Return the hyperparameters, metrics and run_id of the run giving the best value of the given metric.

    'grid_point' is any point of the grid of the model: it gives the names of its
    hyperparameters and the type their values are given with. The metrics are every column of the
    winning row but the hyperparameters and the run_id, cast from numpy to native Python types so
    that they can be written to json."""
    if not metrics_files:
        raise ValueError("No fine-tuning metrics files found.")

    runs = pd.concat(
        [pd.read_csv(f).assign(run_id=i) for i, f in enumerate(metrics_files)],
        ignore_index=True,
    )

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

    hyperparameters = {name: cast_like(best_run[name], reference) for name, reference in grid_point.items()}
    metrics = {
        name: value.item() if hasattr(value, "item") else value
        for name, value in best_run.items()
        if name not in grid_point and name != "run_id"
    }
    return hyperparameters, metrics, int(best_run["run_id"])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Gather the fine-tuning runs of every model and keep the best hyperparameter set of each.")
    argparser.add_argument("--input_dir", required=True, help="Directory holding the fine-tuning metrics of every model.")
    argparser.add_argument("--metrics_file", required=True, help="Output json file for the best hyperparameter set of every model.")
    argparser.add_argument("--model_dir", required=True, help="Directory holding the best models")
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

        grid = grid_of(grids[model])
        metrics_files = [
            os.path.join(args.input_dir, f"{model}", f"{run_id}", "metrics.csv")
            for run_id in range(len(grid))
        ]
        hyperparameters, metrics, run_id = best_hyperparameters(
            metrics_files, grid[0], metric, direction
        )
        best_hyperparameters_of_models[model] = {
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "run_id": run_id,
        }
        print(f"Best hyperparameter set of the model {model} ({direction} {metric}): "
              f"{hyperparameters}")

        ext = model_extension(model)
        model_subdir = os.path.join(args.model_dir, model)
        link_path = os.path.join(model_subdir, f"{model}{ext}")
        target = os.path.join(str(run_id), f"{model}{ext}")  # relative, so the tree stays movable

        if os.path.lexists(link_path):
            os.remove(link_path)  # idempotent on reruns
        os.symlink(target, link_path)

    output_dir = os.path.dirname(args.metrics_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.metrics_file, "w") as f:
        json.dump(best_hyperparameters_of_models, f, indent=0)

