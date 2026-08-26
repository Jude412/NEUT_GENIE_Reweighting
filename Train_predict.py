""" The objective of this script is to create functions training the different reweighting methods (binning, XGB) and giving weights.
This will assume that the input data is given as numpy arrays, already sorted in training and validation sets.
The output will be the trained model, which can be used to predict the weights."""

#imports 
import numpy as np
import pandas as pd
from hep_ml import reweight
from xgboost import XGBClassifier

def train_binning(original_train, target_train, n_bins, n_neighbours, original_train_weight=None, target_train_weight=None):
    """This function trains the binning reweighter given as arguments the training data, the number of bins
    and neighbours, and the pre-weights of the two distributions if they are given as arguments, otherwise it
    will assume that they are all 1s. The output is the trained model, carrying the normalisation ratio between
    the pre-weighted target and the pre-weighted original training samples."""
    if original_train_weight is None:
        original_train_weight = np.ones(original_train.shape[0])
    if target_train_weight is None:
        target_train_weight = np.ones(target_train.shape[0])

    model = reweight.BinsReweighter(n_bins=n_bins, n_neighs=n_neighbours)
    model.fit(original_train, target_train, original_train_weight, target_train_weight)

    shape_weights = model.predict_weights(original_train)
    model.norm_ratio = np.sum(target_train_weight) / np.sum(original_train_weight * shape_weights)
    return model

def predict_binning(model, original):
    """This function returns the multiplier taking the pre-weighted original distribution to the
    pre-weighted target distribution. The pre-weights are not included in the output: multiply by them
    to get the absolute weights."""
    return model.predict_weights(original) * model.norm_ratio

def train_XGB(original_train, original_val, target_train, target_val, 
              hparams = {"n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 1, 
                    "gamma": 1,
                    "early_stopping_rounds": 10}, original_train_weight=None, original_val_weight=None, target_train_weight=None, target_val_weight=None):
    """This function trains an XGBReweighter to reweight the 'original' distribution into the 'target' distribution. 
    The output is a model that can be used to give weights to the 'original' distribution to make it look like the 'target' distribution."""
    if original_train_weight is None:
        original_train_weight = np.ones(original_train.shape[0])
    if target_train_weight is None:
        target_train_weight = np.ones(target_train.shape[0])
    if target_val_weight is None:
        target_val_weight = np.ones(target_val.shape[0])
    if original_val_weight is None:
        original_val_weight = np.ones(original_val.shape[0])
       
    norm_ratio_train = target_train_weight.sum() / original_train_weight.sum()
    norm_ratio_val = target_val_weight.sum() / original_val_weight.sum()
    if abs((norm_ratio_train/norm_ratio_val) - 1) > 0.01:
        print(f"Warning: the normalisation ratio between original and target distributions differs by {abs((norm_ratio_train/norm_ratio_val) - 1):.1%} between the training and validation samples.")

    # --- balance classes for training only (does not touch the ratio above) ---
    original_train_weight_bal = original_train_weight * norm_ratio_train
    original_val_weight_bal = original_val_weight * norm_ratio_val

    X_train = np.concatenate((original_train, target_train), axis=0)
    Y_train = np.concatenate((np.zeros(len(original_train)), np.ones(len(target_train))), axis=0)
    W_train = np.concatenate((original_train_weight_bal, target_train_weight))

    X_val = np.concatenate((original_val, target_val), axis=0)
    Y_val = np.concatenate((np.zeros(len(original_val)), np.ones(len(target_val))), axis=0)
    W_val = np.concatenate((original_val_weight_bal, target_val_weight))

    # Need weights of order 1 for XGB. Since this is only training shape, can just rescale the weights.
    # The median is taken over the events carrying a weight, so that a sample in which more than half of
    # the events have a null weight does not give a null (or non-finite) scale.
    positive_weights = W_train[W_train > 0]
    weight_scale = np.median(positive_weights) if positive_weights.size > 0 else 1.
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        print("Warning: the weights of the training sample give no usable scale: they are left untouched.")
        weight_scale = 1.
    W_train /= weight_scale
    W_val /= weight_scale

    bst = XGBClassifier(objective='binary:logistic',
                    eval_metric=['auc', 'logloss'],
                    **hparams)

    bst.fit(
        X_train, Y_train,
        sample_weight = W_train,
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        sample_weight_eval_set=[W_train, W_val],
        verbose=False
    )

    bst.norm_ratio = norm_ratio_train
    return bst

def best_iteration_of(model):
    """Return the index of the boosting iteration the predictions of a model are taken at.

    'best_iteration' is only defined by XGBoost when early stopping is used, so the last iteration
    of the model is returned when it was trained without 'early_stopping_rounds'."""
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = model.get_booster().num_boosted_rounds() - 1
    return best_iteration

def predict_XGB(original, model):
    """This function uses the trained model to return the weights used for the reweighting process."""
    predictions = model.predict_proba(original, iteration_range=(0, best_iteration_of(model) + 1))
    p = np.clip(predictions[:, 1], 1e-7, 1 - 1e-7)
    shape_weights = p / (1 - p) # Weights for shape matching
    weights = shape_weights * model.norm_ratio # Weights for shape and normalization matching
    return weights
