""" The goal of this script is to create functions that will be used to evaluate the performance of the different reweighting methods.
The idea is to have functions that take as input the original and target distributions, and the predicted weights of any
method and return : a Chi2 statistic, a SWD value and the associated p-values.
We are here trying to implement it with n-dimensional distributions, for which we assume the weights have already been predicted.
The plots displaying those metrics are made by 'Plots_ndim.py'."""

"""In the following functions, there is no need to normalize the weights (ie enforcing np.sum(weights) = 1), 
since the normalization is already included in the computation of the metrics."""

#Imports

import hashlib
import numpy as np
import ot
from scipy.stats import chi2 as chi2_dist
from Param_sets import ALL_PARAMS_SET, REWEIGHTING_SET


def swd_direction_rng(base_seed, param_set):
    """Return the generator drawing the SWD projection directions of a set of parameters.

    The SWD is a Monte-Carlo estimate over random projections, so two jobs drawing their own
    directions disagree on it by a few percent even for identical inputs. Seeding it from the
    sample and topology ('base_seed') makes the baseline job and every hyperparameter job of a
    sample and topology project onto the same directions, so that the hyperparameter sets, and
    the un-reweighted baseline they are read against, differ only by what the reweighting did
    and not by the noise of the estimator. The name of the parameter set is mixed in so that two
    sets holding the same number of parameters do not reuse a single draw.

    'hashlib' rather than the builtin 'hash', which is salted per process and so would not give
    the same seed in the baseline job as in the hyperparameter jobs."""
    set_seed = int(hashlib.sha256(param_set.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([base_seed, set_seed])

def chi2_hist_axis(original, target, rw_weights, axis_number, target_weights = None, n_bins = 30, x_min=None, x_max=None, n_events=None):
    if (x_min is not None) and (x_max is not None):
        bins = np.linspace(x_min, x_max, n_bins+1)
    else:
        first_percentile = np.percentile(target[:, axis_number], 1)
        if np.abs(first_percentile) < 5e-2:
            first_percentile = 0
        ninety_ninth_percentile = np.percentile(target[:, axis_number], 99)
        bins = np.linspace(first_percentile, ninety_ninth_percentile, n_bins+1)

    if target_weights is not None:
        hist_target = np.histogram(target[:, axis_number], bins=bins, weights=target_weights/np.sum(target_weights))[0]
        target_weights_squared_sum = np.histogram(target[:, axis_number], bins=bins, weights=(target_weights/np.sum(target_weights))**2)[0]
    else:
        hist_target = np.histogram(target[:, axis_number], bins=bins, weights=np.ones_like(target[:, axis_number])/len(target))[0]
        target_weights_squared_sum = np.histogram(target[:, axis_number], bins=bins, weights=(np.ones_like(target[:, axis_number])/len(target))**2)[0]

    hist_original_rw = np.histogram(original[:, axis_number], bins=bins, weights=rw_weights/np.sum(rw_weights))[0]

    ## We must adapt the formula to account for the fact we are using weighted events, and that our predicted histogram intrinsically comes with errors
    ## So, we have to sum the errors in quadrature, one with the sqrt of the target histogram, and the other with the sqrt of the sum of the 
    ## squared weights of our prediction in each bin.
    rw_weight_squared_sum = np.histogram(original[:, axis_number], bins=bins, weights=(rw_weights/np.sum(rw_weights))**2)[0]

    # Each term above is a variance that scales as 1/n_events (regardless of the events' weight
    # magnitude, since it is computed from weights already normalised to sum to 1): scaling it by
    # the actual number of events over a hypothetical smaller one reports the chi2 that hypothetical
    # sample would give, rather than this (usually much larger) test sample's.
    target_scale = len(target) / n_events if n_events is not None else 1.0
    original_scale = len(original) / n_events if n_events is not None else 1.0
    sigma2 = target_weights_squared_sum * target_scale + rw_weight_squared_sum * original_scale
    mask = sigma2 > 0
    chi2 = np.sum((hist_original_rw[mask] - hist_target[mask]) ** 2 / (sigma2[mask]))
    if np.sum(mask) >= 2:
        dof = np.sum(mask) - 1
    else :
        dof = 1
    return chi2, dof

def binning_of(param, binning_dict):
    """Return the (n_bins, x_min, x_max) a parameter is histogrammed with: the ones of the binning
    file when it lists the parameter, and 30 bins over the 1-99% range of the target otherwise."""
    if param in binning_dict:
        return binning_dict[param]["n_bins"], binning_dict[param]["x_min"], binning_dict[param]["x_max"]
    return 30, None, None

def chi2_hist_naxis(original, target, rw_weights, binning_dict, target_weights = None, list_param_interest = ["Enu_true", "Plep", "CosLep"], n_events=None):
    chi2 = 0
    dof = 0
    for axis_number, var in enumerate(list_param_interest):
        n_bins, x_min, x_max = binning_of(var, binning_dict)
        chi2_val, dof_val = chi2_hist_axis(original, target, rw_weights, axis_number, target_weights, n_bins=n_bins, x_min=x_min, x_max=x_max, n_events=n_events)
        chi2 += chi2_val
        dof += dof_val
    return chi2, dof

def chi2_dof(original, target, weights_dict, binning_dict, target_weights = None,
             list_param_interest = ["Enu_true", "Plep", "CosLep"], return_chi2_and_dof=False):
    dict_chi2 = {}
    dict_chi2_and_dof = {}
    for key in weights_dict.keys():
        chi2, dof = chi2_hist_naxis(original, target, weights_dict[key], binning_dict, target_weights, list_param_interest)
        dict_chi2_and_dof[key] = (chi2, dof)
        dict_chi2[key] = chi2 / dof if dof > 0 else 0
    if return_chi2_and_dof:
        return dict_chi2_and_dof
    return dict_chi2

def chi2_p_value(chi2, dof):
    p_value = 1 - chi2_dist.cdf(chi2, dof)
    return p_value

def compute_swd(original, target, weights_dict, target_weights = None, n_directions=500, rng=None):
    # We compute the Sliced Wasserstein Distance between the original and target distributions, using the weights for the original distribution.
    # We project the distributions on a given number of random vectors in the unitary sphere, and we compute the wasserstein distance for each projection,
    #then we average them.
    # 'rng' is the generator the directions are drawn from: pass the one 'swd_direction_rng'
    # returns to compare distances across jobs, and leave it unset to draw fresh directions.
    rng = np.random.default_rng() if rng is None else rng
    v = rng.normal(0, 1, size = (n_directions, original.shape[1]))
    v /= np.linalg.norm(v, axis = 1)[:, np.newaxis]

    dict_swd = {key: [] for key in weights_dict.keys()}
    dict_mean_swd = {}

    for vector in v:
        #we project our nd distribution on the random vector, and we compute the wasserstein distance for each reweighting method on the projected distribution.
        #print(original.shape)
        orig_proj = np.dot(original, vector)
        #print(orig_proj.shape)
        target_proj = np.dot(target, vector)
        for k in range(len(weights_dict)):
            key = list(weights_dict.keys())[k]
            w_dist = ot.wasserstein_1d(orig_proj, target_proj, 
                                       u_weights = weights_dict[key]/np.sum(weights_dict[key]), 
                                       v_weights=target_weights/np.sum(target_weights) if target_weights is not None else None)
            dict_swd[key].append(w_dist)
        
    for k in range(len(weights_dict)):
        key = list(weights_dict.keys())[k]
        print(f"Average Wasserstein distance for reweighting with {key}: {np.mean(dict_swd[key])}")
        dict_mean_swd[key] = np.mean(dict_swd[key])
    return dict_mean_swd

"""We then use this function to compute a p-value for the swd we obtained with a given swd distribution 
obtained from bootstraping the target distribution with the Bootstrap_SWD function.  """

def bootstrap_swd(target, n_bootstrap=1000, n_directions=500, sample_size=None):
    """Returns a list of SWD values obtained by bootstraping the target distribution and computing the SWD between the two bootstraped distributions.

    Each of the two bootstrapped distributions holds 'sample_size' events (every event of the target
    when unset), so that the list is the null distribution of the SWD between two independent samples
    of that size: the data-sized subsamples 'compute_all_metrics' compares are read against it."""
    list_swd = []

    target_dist, target_weights = target
    sample_size = target_dist.shape[0] if sample_size is None else sample_size

    for _ in range(n_bootstrap):
        index1 = np.random.choice(target_dist.shape[0], size=sample_size, replace=True)
        index2 = np.random.choice(target_dist.shape[0], size=sample_size, replace=True)
        target_bootstrap_1 = target_dist[index1]
        target_bootstrap_2 = target_dist[index2]

        target_weights_bootstrap_1 = target_weights[index1]
        target_weights_bootstrap_2 = target_weights[index2]

        v = np.random.normal(0, 1, size = (n_directions, target_dist.shape[1]))
        v /= np.linalg.norm(v, axis = 1)[:, np.newaxis]

        list_wass = []

        for vector in v:
            #we project our nd distribution on the random vector, and we compute the wasserstein distance for 
            #each reweighting method on the projected distribution.

            first_proj = np.dot(target_bootstrap_1, vector)
            #print(orig_proj.shape)
            second_proj = np.dot(target_bootstrap_2, vector)
            w_dist = ot.wasserstein_1d(first_proj, second_proj, 
                                       u_weights = target_weights_bootstrap_1/np.sum(target_weights_bootstrap_1),
                                       v_weights = target_weights_bootstrap_2/np.sum(target_weights_bootstrap_2)
                                       )

            list_wass.append(w_dist)

        list_swd.append(np.mean(list_wass))

    return list_swd


def swd_p_value(swd_value, swd_distribution):
    """Fraction of the bootstrapped SWD distribution lying above a given SWD value."""
    return 1 - np.sum(swd_distribution <= swd_value) / len(swd_distribution)

def compute_p_value(swd_value_dict, swd_distribution):
    """Compute the p-value for a given swd value and a swd distribution obtained from bootstraping the target distribution."""
    p_value_dict = {}
    for key in swd_value_dict.keys():
        swd_value = swd_value_dict[key]
        p_value = swd_p_value(swd_value, swd_distribution)
        p_value_dict[key] = p_value
        print(f"P-value for {key}: SWD value {swd_value}: {p_value}")
    return p_value_dict


"""The functions below compute the metrics written by 'Compute_metrics.py' and 'Compute_baseline_metrics.py',
both on the full test samples and at the scale of the real downstream analysis ('data scale'), whose
sample is much smaller than the test samples and so is much less sensitive to a mismodelling."""

def cap_data_n_events(n_events, sample_size):
    """Return the data-scale number of events as an integer, capped at the number of events of the
    test sample(s) it is drawn from: a topology whose test sample is smaller than its data-scale one
    has its data-scale metrics computed on the whole test sample instead."""
    n_events = int(round(n_events))
    if n_events > sample_size:
        print(f"Warning: the data-scale number of events ({n_events}) is larger than the test sample "
              f"({sample_size}): the data-scale metrics are computed on {sample_size} events instead.")
        return sample_size
    return n_events

def data_subsample_indices(n_original, n_target, n_events, n_subsamples, seed):
    """Return 'n_subsamples' pairs of (original, target) event indices, each drawing 'n_events'
    events of the original and of the target test samples without replacement.

    Every parameter set is read from the same files with the same topology filter, so the indices
    apply to the samples of all of them. The baseline job and every 'compute_metrics' job of a sample
    and topology are given the same seed, so that they are all evaluated on the same subsamples and
    the hyperparameter sets differ only by what the reweighting did, not by which events were drawn."""
    rng = np.random.default_rng(seed)
    return [(np.sort(rng.choice(n_original, size=n_events, replace=False)),
             np.sort(rng.choice(n_target, size=n_events, replace=False)))
            for _ in range(n_subsamples)]

def compute_all_metrics(testing_samples, weights, param_set_dict, binning_dict, swd_distributions,
                        swd_seed, n_directions, n_events, subsamples):
    """Return the {metric name: value} dictionary of the original test sample carrying the absolute
    weights 'weights' against the target test sample.

    'testing_samples' maps every parameter set to its {"original_test", "target_test"} samples, and
    'swd_distributions' every parameter set to the npy file of its bootstrapped SWD distribution, drawn
    at the data scale. The metrics are, for every parameter set:
      - 'SWD_{set}': the SWD between the full test samples, along the directions 'swd_seed' gives.
      - 'data_SWD_{set}' and 'data_p_value_{set}': the mean SWD of the data-sized 'subsamples' (as given
        by 'data_subsample_indices'), and the median of their p-values against the bootstrapped SWD
        distribution. A single subsample would make the hyperparameter selection pick the luckiest draw.
      - 'chi2_dof_{set}' and 'chi2_p_value_{set}' on the full test samples, and 'data_chi2_dof_{set}'
        and 'data_chi2_p_value_{set}' with their uncertainties scaled to 'n_events' events.
    and the same chi2 metrics for every analysis parameter taken on its own, plus 'norm_test'."""
    # 'compute_swd' takes a dictionary of weights, of which this is the only key.
    key = "weights"
    # Normalisation closure on the test sample: the summed reweighted original over the summed target,
    # 1 for a correctly normalised model. Every other metric normalises both samples to 1 and so is
    # blind to it.
    metrics = {"norm_test": np.sum(weights) / np.sum(testing_samples[REWEIGHTING_SET]["target_test"][1])}

    for set_name, samples in testing_samples.items():
        original, _ = samples["original_test"]
        target, target_weight = samples["target_test"]

        # A fresh generator is given to every call, so that the full test samples and every subsample
        # are projected along the same directions.
        metrics[f"SWD_{set_name}"] = compute_swd(original, target, {key: weights},
                                                 target_weights=target_weight, n_directions=n_directions,
                                                 rng=swd_direction_rng(swd_seed, set_name))[key]
        data_swds = [compute_swd(original[original_index], target[target_index], {key: weights[original_index]},
                                 target_weights=target_weight[target_index], n_directions=n_directions,
                                 rng=swd_direction_rng(swd_seed, set_name))[key]
                     for original_index, target_index in subsamples]
        metrics[f"data_SWD_{set_name}"] = float(np.mean(data_swds))
        metrics[f"data_p_value_{set_name}"] = float("nan")
        if set_name in swd_distributions:
            swd_distribution = np.load(swd_distributions[set_name])
            metrics[f"data_p_value_{set_name}"] = float(np.median([swd_p_value(swd, swd_distribution)
                                                                   for swd in data_swds]))
        else:
            print(f"Warning: no bootstrapped SWD distribution given for the parameter set '{set_name}': "
                  "its data-scale p-value is not computed.")

        for prefix, scale in (("", None), ("data_", n_events)):
            chi2, dof = chi2_hist_naxis(original, target, weights, binning_dict, target_weight,
                                        param_set_dict[set_name], n_events=scale)
            metrics[f"{prefix}chi2_dof_{set_name}"] = chi2/dof if dof > 0 else 0
            metrics[f"{prefix}chi2_p_value_{set_name}"] = chi2_p_value(chi2, dof)

    # Chi2 of every analysis parameter taken on its own.
    all_params_original, _ = testing_samples[ALL_PARAMS_SET]["original_test"]
    all_params_target, all_params_target_weight = testing_samples[ALL_PARAMS_SET]["target_test"]
    for param_index, param in enumerate(param_set_dict[ALL_PARAMS_SET]):
        n_bins, x_min, x_max = binning_of(param, binning_dict)
        for prefix, scale in (("", None), ("data_", n_events)):
            chi2, dof = chi2_hist_axis(all_params_original, all_params_target, weights,
                                       axis_number=param_index, target_weights=all_params_target_weight,
                                       n_bins=n_bins, x_min=x_min, x_max=x_max, n_events=scale)
            metrics[f"{prefix}chi2_dof_{param}"] = chi2/dof if dof > 0 else 0
            metrics[f"{prefix}chi2_p_value_{param}"] = chi2_p_value(chi2, dof)

    return metrics
