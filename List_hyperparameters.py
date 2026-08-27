"""Write one json file per point of the hyperparameter grid of every model.

The grids are read from a json file (given by the 'models'/'grid_file' section of the config
file) holding, for each model, the list of values every hyperparameter is scanned over:

    {"binning": {"n_bins": [5, 10], "n_neighs": [0, 1]}, "XGB": {...}}

The grid of a model is the cartesian product of those lists, so the example above gives four
sets of hyperparameters. One json file is written per set, named '{model}/{model}_hp_{i}.json',
alongside a '{model}/{model}_job.sub' file listing them for batch submission."""

# imports
from itertools import product
import argparse
import json
import os

def grid_of(model_grid):
    """Return the list of hyperparameter sets of a model, i.e. the cartesian product of its grid."""
    names = list(model_grid)
    value_lists = [model_grid[name] for name in names]
    return [dict(zip(names, values)) for values in product(*value_lists)]

def number_of_sets(grid_file, model):
    """Return the number of hyperparameter sets of a model, as written by this script.

    It is used by the workflow to know how many fine-tuning runs a model needs, so that the
    grid is only ever described in the grid file."""
    with open(grid_file) as f:
        grids = json.load(f)
    return len(grid_of(grids[model]))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Generate the json files of every hyperparameter set of the given models.")
    argparser.add_argument("--grid_file", type=str, required=True,
                           help="Path to the json file holding the hyperparameter grid of every model.")
    argparser.add_argument("--output_dir", type=str, default="hps", help="Directory to save the json files in.")
    argparser.add_argument("--models", nargs="+", required=False,
                           help="Models to write the hyperparameter sets of. Defaults to every model of the grid file.")
    args = argparser.parse_args()

    with open(args.grid_file) as f:
        grids = json.load(f)

    models = args.models if args.models is not None else list(grids)

    for model in models:
        if model not in grids:
            raise ValueError(f"No hyperparameter grid given for the model '{model}' in {args.grid_file}. "
                             f"The grids given are those of {list(grids)}.")

        model_dir = os.path.join(args.output_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        sub_file_lines = []
        for counter, hyperparameters in enumerate(grid_of(grids[model])):
            hp_file = os.path.join(model_dir, f"{model}_hp_{counter}.json")
            with open(hp_file, "w") as f:
                json.dump(hyperparameters, f)
            sub_file_lines.append(os.path.abspath(hp_file))

        with open(os.path.join(model_dir, f"{model}_job.sub"), "w") as f:
            f.write("\n".join(sub_file_lines) + "\n)")

        print(f"Wrote {len(sub_file_lines)} hyperparameter set(s) of the model {model} in {model_dir}.")
