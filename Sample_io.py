"""Reading and writing of the sample csv files used throughout the workflow.

Every sample holds the analysis parameters of the events in its first columns and the total weight of
the events ('PreWeight', the product of the 'RWWeight' and 'fScaleFactor' branches of the input files) in
its last column. The helpers below keep the weights out of the parameters, so that the trainings and the
metrics are carried out on the weighted events without ever using the weight itself as a parameter."""

#imports
import numpy as np
import os

# Name of the column holding the total weight of the events in the sample csv files.
WEIGHT_COLUMN = "PreWeight"

# The samples every set of parameters is split into, as they are named on disk.
SAMPLE_NAMES = ("original_train", "original_val", "original_test",
                "target_train", "target_val", "target_test")

def sample_columns(sample_file):
    """Return the names of the parameters stored in a sample csv file, without the weight column."""
    with open(sample_file) as f:
        header = f.readline()

    columns = [name.strip() for name in header.lstrip("#").strip().split(",")]
    if columns and columns[-1] == WEIGHT_COLUMN:
        columns = columns[:-1]
    return columns


def save_sample(sample_file, distribution, weights, columns):
    """Save a sample as a csv file, with the weights of its events appended as a last column."""
    distribution = np.atleast_2d(distribution)
    weights = np.asarray(weights, dtype=float).reshape(-1, 1)
    np.savetxt(sample_file, np.hstack((distribution, weights)), delimiter=",",
               header=",".join(list(columns) + [WEIGHT_COLUMN]))


def load_sample(sample_file):
    """Load a sample csv file and return its parameters and the weights of its events.

    Samples written without a weight column (for example by an older version of the workflow) are
    given a weight of 1 for every event."""
    data = np.loadtxt(sample_file, delimiter=",", ndmin=2)
    with open(sample_file) as f:
        header = f.readline()

    columns = [name.strip() for name in header.lstrip("#").strip().split(",")]
    if columns and columns[-1] == WEIGHT_COLUMN:
        return data[:, :-1], data[:, -1]

    print(f"Warning: no '{WEIGHT_COLUMN}' column found in {sample_file}: every event is given a weight of 1.")
    return data, np.ones(data.shape[0])

def create_samples(distribution, split_indices, weights = None):
    """Split a distribution into its samples, given the indices of the events of each of them.

    Returns the {sample name: events} dictionary of the samples, and the {sample name: weights}
    dictionary of the weights of their events when weights are given."""
    samples = {name: distribution[indices] for name, indices in split_indices.items()}
    if weights is None:
        return samples

    sample_weights = {name: weights[indices] for name, indices in split_indices.items()}
    return samples, sample_weights

def load_samples(sample_dir):
    """Load every sample of a directory and return the {sample name: (events, weights)} dictionary."""
    return {name: load_sample(os.path.join(sample_dir, f"{name}.csv")) for name in SAMPLE_NAMES}
