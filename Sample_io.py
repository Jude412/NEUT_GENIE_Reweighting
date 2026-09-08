"""Reading and writing of the sample csv files used throughout the workflow.

Every sample holds the analysis parameters of the events in its first columns and the total weight of
the events ('PreWeight', the product of the 'RWWeight' and 'fScaleFactor' branches of the input files) in
its last column. The helpers below keep the weights out of the parameters, so that the trainings and the
metrics are carried out on the weighted events without ever using the weight itself as a parameter."""

#imports
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
from Constants import WEIGHT_COLUMN, TOPOLOGY_COLUMN, topology_code

# The samples every set of parameters is split into, as they are named on disk.
SAMPLE_NAMES = ("original_train", "original_val", "original_test",
                "target_train", "target_val", "target_test")

def save_sample(sample, sample_file):
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
    table = pa.Table.from_pandas(sample)
    pq.write_to_dataset(table,
                        root_path=sample_file,
                        partition_cols=[TOPOLOGY_COLUMN],
                        compression="zstd")

def df_to_np(sample):
    """Convert a sample dataframe to a tuple of (events, weights) numpy arrays."""
    events = sample.drop(columns=[WEIGHT_COLUMN]).to_numpy()
    weights = sample[WEIGHT_COLUMN].to_numpy()
    return events, weights

def sample_params(sample_file):
    """Return the parameter names held by a sample parquet dataset, in schema order, excluding
    the weight column and the topology partition column (which only ever serves to select which
    partition to read, never as a parameter itself)."""
    columns = ds.dataset(sample_file, partitioning="hive").schema.names
    assert WEIGHT_COLUMN in columns, f"'{WEIGHT_COLUMN}' column not found in {sample_file}"
    return [column for column in columns if column not in (WEIGHT_COLUMN, TOPOLOGY_COLUMN)]

def load_sample(sample_file, topology=None, params=None):
    """Load a sample parquet file and return the listed parameters with weights"""
    if params is None:
        params = sample_params(sample_file)
    columns_to_read = list(params) + [WEIGHT_COLUMN]

    if topology is None:
        return df_to_np(pd.read_parquet(sample_file, columns=columns_to_read))
    else:
        return df_to_np(pd.read_parquet(sample_file,
                             columns=columns_to_read,
                             filters=[(TOPOLOGY_COLUMN, "==", topology_code(topology))]))

def split_sample(sample, train_percentage, val_percentage, included_topologies, random_seed):
    """Split a distribution into its samples, given the indices of the events of each of them.

    Returns three dataframes: train_df, val_df, test_df"""
    train_parts, val_parts, test_parts = [], [], []
    rng = np.random.default_rng(random_seed)
    
    for topology, group in sample.groupby(TOPOLOGY_COLUMN):
        if topology not in included_topologies:
            continue

        idx = rng.permutation(len(group))
        group = group.iloc[idx]

        n = len(group)
        n_train = int(train_percentage * n)
        n_val = int(val_percentage * n)
        n_test = n - n_train - n_val

        assert n_train >= 1 and n_val >= 1 and n_test >= 1, (
            f"Topology {topology} has too few events ({n}) to be split into training, validation and test samples with the requested percentages "
        )

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df

def load_samples(sample_dir, topology=None, params=None, sample_names=SAMPLE_NAMES):
    """Load every sample of a directory and return the {sample name: (events, weights)} dictionary."""
    return {name: load_sample(os.path.join(sample_dir, f"{name}.parquet"), topology=topology, params=params) for name in sample_names}
