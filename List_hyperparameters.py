"""Write one json file per point of the hyperparameter grid of every model.

The grids are read from a json file (given by the 'models'/'grid_file' section of the config
file) holding, for each model, the list of values every hyperparameter is scanned over:

    {"binning": {"n_bins": [5, 10], "n_neighs": [0, 1]}, "XGB": {...}}

The grid of a model is the cartesian product of those lists, so the example above gives four
sets of hyperparameters. A model can instead be given a list of such grids, to scan the union of
their sets rather than every combination of their values:
    {"XGB": [{"learning_rate": [0.1], "gamma_frac": [6e-6]}, {"learning_rate": [0.3], "gamma_frac": [0]}]}
gives two sets. The grids should give the same hyperparameters, so that every set is labelled alike,
and no two sets may be the same. One json file is written per set, named
'{model}/{model}_hp_{label}.json', alongside a '{model}/{model}_job.sub' file listing them for
batch submission."""


# imports
from itertools import product
import argparse
import json
import os
import re

# The characters a value is allowed to contribute to a label. Anything else (a path separator,
# a space) is dropped, so that a label is always a single, well-behaved directory name. Only
# string-valued hyperparameters can hold such characters; numbers never do.
UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9.+-]")

def format_value(value):
    """Return the string a hyperparameter value is labelled with."""

    if isinstance(value, bool):  # before the int branch, as a bool is an int
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "None"
    return UNSAFE_CHARACTERS.sub("", str(value))

def label_of(hyperparameters):
    """Return the label a set of hyperparameters is named with, ex 'n_bins-10_n_neighs-1'.

    The hyperparameters are sorted by name, so that the label of a set depends on the set alone:
    reordering the grid file does not rename anything."""
    return "_".join(f"{name}-{format_value(hyperparameters[name])}"
                    for name in sorted(hyperparameters))

def grid_of(model_grid):
    """Return the list of hyperparameter sets of a model, i.e. the cartesian product of its grid, or,
    when it is given a list of grids, the sets of every one of them in turn."""
    if isinstance(model_grid, list):
        return [hyperparameters for grid in model_grid for hyperparameters in grid_of(grid)]
    names = list(model_grid)
    value_lists = [model_grid[name] for name in names]
    return [dict(zip(names, values)) for values in product(*value_lists)]

def labelled_grid_of(model_grid):
    """Return the {label: hyperparameters} dictionary of the grid of a model."""

    labelled = {}
    for hyperparameters in grid_of(model_grid):
        label = label_of(hyperparameters)
        if label in labelled:
            raise ValueError(f"The hyperparameter sets {labelled[label]} and {hyperparameters} of "
                             f"the grid are both labelled '{label}', and so would share every "
                             f"output directory. Please give the values of the grid that label the "
                             f"same (ex: 1 and 1.0) only once.")
        labelled[label] = hyperparameters
    return labelled

def grids_of_file(grid_file, model):
    """Return the grid a model is scanned over, read from the grid file."""
    with open(grid_file) as f:
        grids = json.load(f)
    if model not in grids:
        raise ValueError(f"No hyperparameter grid given for the model '{model}' in {grid_file}. "
                         f"The grids given are those of {list(grids)}.")
    return grids[model]

def labels_of(grid_file, model):
    """Return the label of every hyperparameter set of a model, as written by this script.

    It is used by the workflow to know which fine-tuning runs a model needs, so that the grid is
    only ever described in the grid file."""
    return list(labelled_grid_of(grids_of_file(grid_file, model)))

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
        model_grid = grids_of_file(args.grid_file, model)

        model_dir = os.path.join(args.output_dir, model)
        os.makedirs(model_dir, exist_ok=True)

        sub_file_lines = []
        for label, hyperparameters in labelled_grid_of(model_grid).items():
            hp_file = os.path.join(model_dir, f"{model}_hp_{label}.json")
            with open(hp_file, "w") as f:
                json.dump(hyperparameters, f)
            sub_file_lines.append(os.path.abspath(hp_file))

        with open(os.path.join(model_dir, f"{model}_job.sub"), "w") as f:
            f.write("\n".join(sub_file_lines) + "\n)")

        print(f"Wrote {len(sub_file_lines)} hyperparameter set(s) of the model {model} in {model_dir}.")
