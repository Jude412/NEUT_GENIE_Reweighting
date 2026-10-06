"""The goal of this script is train and save a reweighting model for one model and set of hyperparameters."""

#imports
from Sample_io import load_samples
from Train_predict import hyperparameters_of, predict_model, train_model, save_model
import os
import argparse
import json
import numpy as np

def int_or_none(value):
    """Cast a command line argument to int, except for 'None' (any case), which is left as None.

    An optional config option that is left unset reaches the scripts as the string 'None', the
    workflow having no way to leave the argument out, so it is read back as one here. This mirrors
    'float_or_none' of 'Init.py', which does the same for the downsampling fraction."""
    return None if value.lower() == "none" else int(value)

def subsample(events, weights, fraction, rng):
    """Return a random fraction of the events of a sample, and their weights."""
    n = len(events)
    n_keep = int(round(n * fraction))
    idx = rng.choice(n, size=n_keep, replace=False)
    return events[idx], weights[idx]

def cap_paired_samples(samples, pairs, max_events, rng):
    """Downsample each (a, b) pair in `pairs` so the larger of the two has
    at most max_events, scaling both by the same fraction. Modifies and
    returns `samples` in place."""
    if max_events is None:
        return samples

    for name_a, name_b in pairs:
        n_a = len(samples[name_a][0])
        n_b = len(samples[name_b][0])
        larger = max(n_a, n_b)
        if larger <= max_events:
            continue
        fraction = max_events / larger
        samples[name_a] = subsample(*samples[name_a], fraction, rng)
        samples[name_b] = subsample(*samples[name_b], fraction, rng)

    return samples

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a reweighting model with one set of hyperparameters and evaluate it.")
    argparser.add_argument('--sample_dir', type=str, required=True,
                           help='Directory where the training original and target samples are stored in parquet format.')
    argparser.add_argument('--model', type=str, required=True, help='The reweighting model to train.')
    argparser.add_argument('--topology', type=str, required=True, help='The topology to filter the training samples on.')
    argparser.add_argument('--hyperparameters', type=str, required=True, help="Path to the JSON file containing the hyperparameters to train the model with.")
    argparser.add_argument("--reweighting_params", nargs="+", required=True, help="List of parameters to use for the reweighting.")
    argparser.add_argument("--output_dir", type=str, required=True, help="The path to the output directory for the model.")
    argparser.add_argument("--max_events", type=int_or_none, default=None,
                           help="Maximum number of events to train on, per sample. The original and "
                                "target samples are both scaled by the fraction bringing the larger "
                                "of the two down to it, so that their normalisation ratio is kept. "
                                "If None, every event is used.")
    argparser.add_argument("--seed", type=int_or_none, default=None,
                           help="Seed the training sample is capped with. Give the same seed to every "
                                "hyperparameter set of a sample and topology, so that the runs of a "
                                "scan differ by their hyperparameters and not by the events they were "
                                "given. If None, a different subset is drawn every run.")
    args = argparser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load the training and validation samples for the reweighting parameter set
    training_samples = load_samples(args.sample_dir, topology=args.topology, params=args.reweighting_params, sample_names=("original_train", "target_train", "original_val", "target_val"))

    # Cap the samples such that none have more than max_events
    cap_paired_samples(
        training_samples,
        pairs=[("original_train", "target_train"), ("original_val", "target_val")],
        max_events=args.max_events,
        rng=rng,
    )

    # Train the model
    hyperparams = hyperparameters_of(args.model, args.hyperparameters)
    model, resolved_hyperparams = train_model(args.model, training_samples, hyperparams)
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{args.model}")
    save_model(args.model, model, model_path)

    # Written alongside the model rather than stored on it, so that 'Compute_metrics.py' can log the
    # absolute hyperparameters the model was actually trained with (a '..._frac' hyperparameter is
    # resolved to a value that depends on the training sample's size, see
    # 'Train_predict.resolve_fractional_hyperparameters') without needing to reload or retrain it.
    resolved_hyperparams_path = os.path.join(args.output_dir, f"{args.model}_resolved_hyperparameters.json")
    with open(resolved_hyperparams_path, "w") as f:
        json.dump(resolved_hyperparams, f)

    print(f"{args.model} model saved at {model_path}")
