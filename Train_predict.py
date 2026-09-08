""" The objective of this script is to create functions training the different reweighting methods (binning, XGB) and giving weights.
This will assume that the input data is given as numpy arrays, already sorted in training and validation sets.
The output will be the trained model, which can be used to predict the weights."""

#imports
import json
import pickle
import numpy as np
from hep_ml import reweight
from xgboost import XGBClassifier

# The reweighting models that can be trained, with the hyperparameters they are trained with when
# none are given.
DEFAULT_HYPERPARAMETERS = {
    "binning": {"n_bins": 12, "n_neighs": 0},
    "XGB": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "gamma": 2,
            "subsample": 0.3, "early_stopping_rounds": 10},
    "unnormXGB": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "gamma": 2,
            "subsample": 0.3, "early_stopping_rounds": 10},
}
MODEL_NAMES = tuple(DEFAULT_HYPERPARAMETERS)

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

    x_train = np.concatenate((original_train, target_train), axis=0)
    y_train = np.concatenate((np.zeros(len(original_train)), np.ones(len(target_train))), axis=0)
    w_train = np.concatenate((original_train_weight_bal, target_train_weight))

    x_val = np.concatenate((original_val, target_val), axis=0)
    y_val = np.concatenate((np.zeros(len(original_val)), np.ones(len(target_val))), axis=0)
    w_val = np.concatenate((original_val_weight_bal, target_val_weight))

    # Need weights of order 1 for XGB. Since this is only training shape, can just rescale the weights.
    # The median is taken over the events carrying a weight, so that a sample in which more than half of
    # the events have a null weight does not give a null (or non-finite) scale.
    positive_weights = w_train[w_train > 0]
    weight_scale = np.median(positive_weights) if positive_weights.size > 0 else 1.
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        print("Warning: the weights of the training sample give no usable scale: they are left untouched.")
        weight_scale = 1.
    w_train /= weight_scale
    w_val /= weight_scale

    bst = XGBClassifier(objective='binary:logistic',
                    eval_metric=['auc', 'logloss'],
                    **hparams)

    bst.fit(
        x_train, y_train,
        sample_weight = w_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
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

def train_unnormXGB(original_train, original_val, target_train, target_val, 
              hparams = {"n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 1, 
                    "gamma": 1,
                    "early_stopping_rounds": 10}, original_train_weight=None, original_val_weight=None, target_train_weight=None, target_val_weight=None):
    """This function trains an XGBReweighter to reweight the 'original' distribution into the 'target' distribution, without normalising the classes. 
    The output is a model that can be used to give weights to the 'original' distribution to make it look like the 'target' distribution."""
    if original_train_weight is None:
        original_train_weight = np.ones(original_train.shape[0])
    if target_train_weight is None:
        target_train_weight = np.ones(target_train.shape[0])
    if target_val_weight is None:
        target_val_weight = np.ones(target_val.shape[0])
    if original_val_weight is None:
        original_val_weight = np.ones(original_val.shape[0])
       
    x_train = np.concatenate((original_train, target_train), axis=0)
    y_train = np.concatenate((np.zeros(len(original_train)), np.ones(len(target_train))), axis=0)
    w_train = np.concatenate((original_train_weight, target_train_weight))

    x_val = np.concatenate((original_val, target_val), axis=0)
    y_val = np.concatenate((np.zeros(len(original_val)), np.ones(len(target_val))), axis=0)
    w_val = np.concatenate((original_val_weight, target_val_weight))

    # Need weights of order 1 for XGB. Can just rescale all the weights.
    # The median is taken over the events carrying a weight, so that a sample in which more than half of
    # the events have a null weight does not give a null (or non-finite) scale.
    positive_weights = w_train[w_train > 0]
    weight_scale = np.median(positive_weights) if positive_weights.size > 0 else 1.
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        print("Warning: the weights of the training sample give no usable scale: they are left untouched.")
        weight_scale = 1.
    w_train /= weight_scale
    w_val /= weight_scale

    bst = XGBClassifier(objective='binary:logistic',
                    eval_metric=['auc', 'logloss'],
                    **hparams)

    bst.fit(
        x_train, y_train,
        sample_weight = w_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        verbose=False
    )

    return bst

def predict_unnormXGB(original, model):
    """This function uses the trained model to return the weights used for the reweighting process."""
    predictions = model.predict_proba(original, iteration_range=(0, best_iteration_of(model) + 1))
    p = np.clip(predictions[:, 1], 1e-7, 1 - 1e-7)
    weights = p / (1 - p) # Weights for shape and norm matching
    return weights

def check_model(model_name):
    """Raise if the given model is not one of the models that can be trained."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Invalid model choice '{model_name}'. Please choose from {list(MODEL_NAMES)}.")

def model_extension(model_name):
    """Return the file extension a model is saved with by 'save_model'."""
    check_model(model_name)
    return ".pkl" if model_name == "binning" else ".json"

def hyperparameters_of(model_name, hyperparameters_file=None):
    """Return the hyperparameters a model is to be trained with.

    They are read from the given json file, which holds either the hyperparameters of the model
    itself (as written by 'List_hyperparameters.py') or the {model: hyperparameters} dictionary of
    several models (as written by 'Gather_fine_tuning.py'). The default hyperparameters of the
    model are used when no file is given."""
    check_model(model_name)
    if hyperparameters_file is None:
        return dict(DEFAULT_HYPERPARAMETERS[model_name])

    with open(hyperparameters_file) as f:
        hyperparameters = json.load(f)
    return dict(hyperparameters.get(model_name, hyperparameters))

def train_model(model_name, samples, hyperparameters):
    """Train a model on the samples loaded by 'Sample_io.load_samples'.

    The dispatch between the models lives here, so that every script trains them the same way."""
    check_model(model_name)

    # extract numpy arrays from the dataframes, and the pre-weights of the events
    original_train, original_train_weight = samples["original_train"]
    original_val, original_val_weight = samples["original_val"]
    target_train, target_train_weight = samples["target_train"]
    target_val, target_val_weight = samples["target_val"]

    if model_name == "binning":
        return train_binning(original_train, target_train,
                             hyperparameters["n_bins"], hyperparameters["n_neighs"],
                             original_train_weight=original_train_weight,
                             target_train_weight=target_train_weight)

    elif model_name == "XGB":
        return train_XGB(original_train, original_val, target_train, target_val, hparams=hyperparameters,
                         original_train_weight=original_train_weight, original_val_weight=original_val_weight,
                         target_train_weight=target_train_weight, target_val_weight=target_val_weight)

    return train_unnormXGB(original_train, original_val, target_train, target_val, hparams=hyperparameters,
                           original_train_weight=original_train_weight, original_val_weight=original_val_weight,
                           target_train_weight=target_train_weight, target_val_weight=target_val_weight)

def predict_model(model_name, model, original):
    """Return the weights a trained model gives to a distribution.

    They are the multipliers taking the pre-weighted original distribution to the pre-weighted
    target one: multiply them by the pre-weights of the events to get their absolute weights."""
    check_model(model_name)
    if model_name == "binning":
        return predict_binning(model, original[0])
    elif model_name == "XGB":
        return predict_XGB(original[0], model)
    return predict_unnormXGB(original[0], model)

def save_model(model_name, model, path):
    """Save a trained model to the given path (without extension), picking the serialization
    format best suited to it.

    'XGB' and 'unnormXGB' are saved in XGBoost's own model format, which stays readable across
    XGBoost versions, unlike a pickled 'XGBClassifier'. The normalisation ratio 'XGB' carries as
    an extra attribute, and the 'evals_result' training history, are not part of that format by
    default, so they are stored as booster attributes beforehand, which XGBoost does persist.
    'binning' has no such native format and is pickled instead."""
    check_model(model_name)
    if model_name == "binning":
        with open(path + ".pkl", "wb") as f:
            pickle.dump(model, f)
        return

    if model_name == "XGB":
        model.get_booster().set_attr(norm_ratio=str(model.norm_ratio))
    model.get_booster().set_attr(evals_result=json.dumps(model.evals_result()))
    model.save_model(path + ".json")

def load_model(model_name, path):
    """Load a model saved by 'save_model' from the given path (without extension)."""
    check_model(model_name)
    if model_name == "binning":
        with open(path + ".pkl", "rb") as f:
            return pickle.load(f)

    model = XGBClassifier()
    model.load_model(path + ".json")
    if model_name == "XGB":
        norm_ratio_attr = model.get_booster().attr("norm_ratio")
        if norm_ratio_attr is not None:
            model.norm_ratio = float(norm_ratio_attr)
    evals_result_attr = model.get_booster().attr("evals_result")
    if evals_result_attr is not None:
        model.evals_result_ = json.loads(evals_result_attr)
    return model
