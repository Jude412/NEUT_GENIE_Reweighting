"""The objective of this script is to create training and validation samples of the n-dimensional distributions that will
be used to train the reweighting techniques and validate their performance. The percentage of the training and validation 
samples is determined by the user. The function returns the samples as numpy arrays."""

#imports
import numpy as np

def create_samples(distribution, percentage_train, percentage_val, random_seed = 42, weights = None):
    if distribution.shape[0] == 0:
        raise ValueError(
            "Cannot create samples from an empty distribution (0 events). "
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
