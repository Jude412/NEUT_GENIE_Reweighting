""" The goal of this script is to create functions that will be used to evaluate the performance of the different reweighting methods.
The idea is to have functions that take as input the original and target distributions, and the predicted weights of any
method and return : a Chi2 statistic, a SWD value and the associated p-values.
We are here trying to implement it with n-dimensional distributions, for which we assume the weights have already been predicted.
The plots displaying those metrics are made by 'Plots_ndim.py'."""

"""In the following functions, there is no need to normalize the weights (ie enforcing np.sum(weights) = 1), 
since the normalization is already included in the computation of the metrics."""

#Imports

import numpy as np
import ot
from scipy.stats import chi2 as chi2_dist

def chi2_hist_axis(original, target, rw_weights, axis_number, target_weights = None, n_bins = 30, x_min=None, x_max=None):
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
    sigma2 = target_weights_squared_sum + rw_weight_squared_sum
    mask = sigma2 > 0
    chi2 = np.sum((hist_original_rw[mask] - hist_target[mask]) ** 2 / (sigma2[mask]))
    if np.sum(mask) >= 2:
        dof = np.sum(mask) - 1
    else :
        dof = 1
    return chi2, dof

def chi2_hist_naxis(original, target, rw_weights, binning_dict, target_weights = None, list_param_interest = ["Enu_true", "Plep", "CosLep"]):
    n = original.shape[1]
    chi2 = 0
    dof = 0
    for var in list_param_interest:
        axis_number = list_param_interest.index(var)
        if var in binning_dict.keys():
            x_min = binning_dict[var]["x_min"]
            x_max = binning_dict[var]["x_max"]
            n_bins = binning_dict[var]["n_bins"]
        else :
            x_min = None
            x_max = None
            n_bins = 30
        chi2_val, dof_val = chi2_hist_axis(original, target, rw_weights, axis_number, target_weights, n_bins=n_bins, x_min=x_min, x_max=x_max)
        chi2 += chi2_val
        dof += dof_val
    return chi2, dof

def chi2_dof(original, target, weights_dict, binning_dict, target_weights = None, list_param_interest = ["Enu_true", "Plep", "CosLep"]):
    dict_chi2 = {}
    for key in weights_dict.keys():
        chi2, dof = chi2_hist_naxis(original, target, weights_dict[key], binning_dict, target_weights, list_param_interest)
        dict_chi2[key] = chi2 / dof if dof > 0 else 0
    return dict_chi2

def chi2_p_value(chi2, dof):
    p_value = 1 - chi2_dist.cdf(chi2, dof)
    return p_value

def compute_swd(original, target, weights_dict, target_weights = None, n_directions=500):
    #we compute the Sliced Wasserstein Distance between the original and target distributions, using the weights for the original distribution.
    #we project the distributions on a given number of random vectors in the unitary sphere, and we compute the wasserstein distance for each projection,
    #then we average them.
    v = np.random.normal(0, 1, size = (n_directions, original.shape[1]))
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

def bootstrap_swd(target, n_bootstrap=1000, n_directions=500, target_weights = None):
    """Returns a list of SWD values obtained by bootstraping the target distribution and computing the SWD between the two bootstraped distributions."""
    list_swd = []

    for _ in range(n_bootstrap):
        index1 = np.random.choice(target.shape[0], size=target.shape[0], replace=True)
        index2 = np.random.choice(target.shape[0], size=target.shape[0], replace=True)
        target_bootstrap_1 = target[index1]
        target_bootstrap_2 = target[index2]

        if target_weights is not None:
            target_weights_bootstrap_1 = target_weights[index1]
            target_weights_bootstrap_2 = target_weights[index2]

        v = np.random.normal(0, 1, size = (n_directions, target.shape[1]))
        v /= np.linalg.norm(v, axis = 1)[:, np.newaxis]

        list_wass = []

        for vector in v:
            #we project our nd distribution on the random vector, and we compute the wasserstein distance for 
            #each reweighting method on the projected distribution.
            
            first_proj = np.dot(target_bootstrap_1, vector)
            #print(orig_proj.shape)
            second_proj = np.dot(target_bootstrap_2, vector)
            w_dist = ot.wasserstein_1d(first_proj, second_proj, 
                                       u_weights = target_weights_bootstrap_1/np.sum(target_weights_bootstrap_1) if target_weights is not None else np.ones_like(target_bootstrap_1[:, 0])/len(target_bootstrap_1),
                                       v_weights = target_weights_bootstrap_2/np.sum(target_weights_bootstrap_2) if target_weights is not None else np.ones_like(target_bootstrap_2[:, 0])/len(target_bootstrap_2))

            list_wass.append(w_dist)
        
        list_swd.append(np.mean(list_wass))
        
    return list_swd



def compute_p_value(swd_value_dict, swd_distribution):
    """Compute the p-value for a given swd value and a swd distribution obtained from bootstraping the target distribution."""
    p_value_dict = {}
    for key in swd_value_dict.keys():
        swd_value = swd_value_dict[key]
        p_value = 1 - np.sum(swd_distribution <= swd_value) / len(swd_distribution)
        p_value_dict[key] = p_value
        print(f"P-value for {key}: SWD value {swd_value}: {p_value}")
    return p_value_dict
