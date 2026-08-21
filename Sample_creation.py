"""The objective of this script is to create training and validation samples of the n-dimensional distributions that will
be used to train the reweighting techniques and validate their performance. The percentage of the training and validation 
samples is determined by the user. The function returns the samples as numpy arrays."""

#imports
import numpy as np

from split_sizes import minimum_events

def create_samples(distribution, percentage_train, percentage_val, random_seed = 42, weights = None):
    n_events = distribution.shape[0]
    min_events = minimum_events(percentage_train, percentage_val)
    if n_events < min_events:
        raise ValueError(
            f"Cannot split a distribution of {n_events} event(s) into a non-empty training, "
            f"validation and test sample with a training percentage of {percentage_train} and a "
            f"validation percentage of {percentage_val}: at least "
            f"{min_events} events are needed. "
            "Check your sample selection and mode filters."
        )

    indices = np.arange(distribution.shape[0])
    np.random.seed(random_seed)
    train_idx = np.random.choice(indices, size=int(percentage_train*len(distribution)), replace=False)
    all_but_train_idx = np.setdiff1d(indices, train_idx)
    val_indices = np.random.choice(all_but_train_idx, size=int(percentage_val*len(distribution)), replace=False)
    test_indices = np.setdiff1d(all_but_train_idx, val_indices)
    train_sample = distribution[train_idx]
    val_sample = distribution[val_indices]
    test_sample = distribution[test_indices]
    if weights is not None:
        train_weights = weights[train_idx]
        val_weights = weights[val_indices]
        test_weights = weights[test_indices]
        return train_sample, val_sample, test_sample, train_weights, val_weights, test_weights
    else:
        return train_sample, val_sample, test_sample
