""" The goal of this script is to create functions that will be used to evaluate the performance of the different reweighting methods. 
The idea is to have functions that take as input the original and target distributions, and the predicted weights of any
method and return : 1D and 2D histograms with ratio of reweighting data over target, a Chi2 statistics, a SWD value and the associated p-value.
We are here trying to implement it with n-dimensional distributions, for which we assume the weights have already been predicted."""

"""In the following functions, there is no need to normalize the weights (ie enforcing np.sum(weights) = 1), 
since the normalization is already included in the computation of the metrics and in the plots."""

#Imports

import numpy as np
import matplotlib.pyplot as plt
import ot
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as mh
from scipy.stats import chi2 as chi2_dist

def plot_histograms(original, target, weights_dict, dict_binning, original_weights = None, target_weights = None, 
                    xlabels = ["E_nu(GeV)", "E_lepton(GeV)", "cos_theta_lepton"],
                    variables = ["E_nu", "E_lepton", "cos_theta_lepton"],
                    add_wass_distance = True, add_chi2 = True, output_file = "/vols/dune/jmm224/t2knova/reweighting/saved_figures/live.pdf"):
    """This function plots the original, reweighted and target distributions for each variable, with a ratio plot of the reweighted
      distribution over the target distribution. The weights_dict is a dictionary containing the predicted weights for each method, 
      with the method name as key and the weights as value."""
    mh.style.use("DUNE")
    with PdfPages(f"{output_file}") as pdf:
        for var in variables:    
            #For the sake of the plots in this function, we use a given binning for each variable, specified in the dict_binning.
            #If the variable is not present, we use a default binning : uniform between the 1st and 99th percentiles of the target distribution for this 
            #variable.
            i = variables.index(var)
            if var in dict_binning.keys():
                x_min = dict_binning[var]["x_min"]
                x_max = dict_binning[var]["x_max"]
                n_bins = dict_binning[var]["n_bins"]
                bins = np.linspace(x_min, x_max, n_bins)
            else:
                x_min = None
                x_max = None
                first_percentile = np.percentile(target[:, i], 1)
                if np.abs(first_percentile) < 5e-2:
                    first_percentile = 0
                ninety_ninth_percentile = np.percentile(target[:, i], 99)
                bins = np.linspace(first_percentile, ninety_ninth_percentile, 31)

            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Getting the distributions and their uncertainties, with a normalization to the total number of events
            if target_weights is not None:
                target_counts = np.histogram(target[:, i], bins=bins, weights = target_weights/np.sum(target_weights))[0]
                target_counts_uncert = np.histogram(target[:, i], bins=bins, weights= (target_weights/np.sum(target_weights))**2)[0]
            else:
                target_counts = np.histogram(target[:, i], bins=bins, weights = np.ones_like(target[:, i])/len(target))[0]
                target_counts_uncert = np.histogram(target[:, i], bins=bins, weights= (np.ones_like(target[:, i])/len(target))**2)[0]

            if original_weights is not None:
                original_counts = np.histogram(original[:, i], bins=bins, weights = original_weights/np.sum(original_weights))[0]
                orig_counts_uncert = np.histogram(original[:, i], bins=bins, weights= (original_weights/np.sum(original_weights))**2)[0]
            else:
                original_counts = np.histogram(original[:, i], bins=bins, weights = np.ones_like(original[:, i])/len(original))[0]
                orig_counts_uncert = np.histogram(original[:, i], bins=bins, weights= (np.ones_like(original[:, i])/len(original))**2)[0]

            target_uncertainty = np.sqrt(target_counts_uncert)
            orig_uncertainty = np.sqrt(orig_counts_uncert)

            # Getting the distribution for the reweighted original distributions, and its uncertainty (propagated from the weights uncertainty)
        
            original_rw_counts_list = []
            original_rw_counts_uncert_list = []
        
            for k in range(len(weights_dict)):
                key = list(weights_dict.keys())[k]
                original_rw_counts, _ = np.histogram(original[:, i], bins=bins, weights= weights_dict[key]/np.sum(weights_dict[key]))
                original_rw_counts_uncert = np.histogram(original[:, i], bins=bins, weights= (weights_dict[key]/np.sum(weights_dict[key]))**2)[0]
                original_rw_uncertainty = np.sqrt(original_rw_counts_uncert)
                original_rw_counts_list.append(original_rw_counts)
                original_rw_counts_uncert_list.append(original_rw_uncertainty)


            # print(np.sum(target_counts))
            # print(np.sum(original_counts))
            
            fig, (ax_main, ax_ratio) = plt.subplots(
                2, 1, figsize=(8, 8), gridspec_kw={'height_ratios':[2,1], 'hspace': 0.1}, sharex=True
            )

            colors = {}
            markers_list = ['s', 'd', '^', 'p', '*', 'h']  # Add more markers if needed
            chi2_values = {}

            #  Metrics
            text_str = "WD- "
            if add_wass_distance :
                for key in weights_dict.keys():
                    wass_distance = ot.wasserstein_1d(original[:, i], target[:, i], 
                                                u_weights = weights_dict[key]/np.sum(weights_dict[key]),
                                                v_weights = target_weights/np.sum(target_weights) if target_weights is not None else None)
                    text_str += f"{key}: {wass_distance:.2g}, "

            if add_chi2:
                for key in weights_dict.keys():
                    chi2, dof = chi2_hist_axis(original, target, weights_dict[key], i, target_weights, n_bins= 30, x_min=x_min, x_max=x_max)
                    chi2_values[key] = chi2/dof if dof > 0 else 0

            # if text_str != "WD- \nChi2-":
            #     mh.add_text(text_str, ax = ax_main, loc = "over right", fontsize = 6)
                # ax_main.text(0.95, 0.95, text_str, transform=ax_main.transAxes, fontsize=5,
                #         verticalalignment='top', horizontalalignment='right',
                #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Distribution plot
            ax_main.errorbar(bin_centers, target_counts, yerr=target_uncertainty, fmt='o', label='Target')
            ax_main.set_xlim(bins[0], bins[-1])
        
            for k in range(len(weights_dict)):
                key = list(weights_dict.keys())[k]
                line_rw, = ax_main.step(bins, np.r_[original_rw_counts_list[k],
                                        original_rw_counts_list[k][-1]], 
                                        markersize=0,
                                        where='post',
                                        label=f'{key[0].upper() + key[1:]}' + r" $\chi^2_{dof}$:" + f" {chi2_values[key]:.2f}" if add_chi2 else f'{key[0].upper() + key[1:]}')
                colors[f'{key}'] = line_rw.get_color()
                
                original_rw_unc_lower = np.r_[original_rw_counts_list[k] - original_rw_counts_uncert_list[k], (original_rw_counts_list[k] - original_rw_counts_uncert_list[k])[-1]]
                original_rw_unc_upper = np.r_[original_rw_counts_list[k] + original_rw_counts_uncert_list[k], (original_rw_counts_list[k] + original_rw_counts_uncert_list[k])[-1]]

                ax_main.fill_between(bins, original_rw_unc_lower, 
                                    original_rw_unc_upper, step = 'post', 
                                    color = colors[f'{key}'],
                                    alpha=0.3)
            bottom, top = ax_main.get_ylim()
            ax_main.set_ylim(int(0), 1.2*top)
            ax_main.set_ylabel("Frequency", fontsize=22)
            # handles, labels = ax_main.get_legend_handles_labels()
            # order = [0, 4, 2, 3, 1, 5]  
            ax_main.legend(fontsize = 15, ncols = 3, loc = 'upper center', handlelength=1.2)
        

            # Ratio plot (Data / Reweighted Original)

            for k in range(len(weights_dict)):
                ratio = target_counts / original_rw_counts_list[k]

                # Reweighting uncertainty for the points (propagated to ratio)
                ratio_rw_err = original_rw_counts_uncert_list[k] / original_rw_counts_list[k]

                # Plot ratio points with reweighting uncertainty
                ax_ratio.step(bins, np.r_[ratio, ratio[-1]],
                              where = 'post', 
                              markersize = 0.3, 
                              marker = markers_list[k], 
                              color = colors[f'{list(weights_dict.keys())[k]}'])
                ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_rw_err, fmt=markers_list[k], capsize=3, color = colors[f'{list(weights_dict.keys())[k]}'])

            # Data uncertainty for the y=1 band
            rel_unc = np.zeros_like(target_uncertainty, dtype=float)

            np.divide(
                target_uncertainty,
                target_counts,
                out=rel_unc,
                where=target_counts != 0
            )
            
            data_band_lower = np.r_[1 - rel_unc, (1 - rel_unc)[-1]]
            data_band_upper = np.r_[1 + rel_unc, (1 + rel_unc)[-1]]

            # Plot horizontal line y=1 with data uncertainty band
            ax_ratio.fill_between(bins, data_band_lower, data_band_upper, color='grey', alpha=0.3, step='post', label='Stat unc.')
            ax_ratio.set_xlim(bins[0], bins[-1])
            ax_ratio.axhline(1, color='gray', linestyle='--')

            ax_ratio.set_xlabel(xlabels[i], fontsize=22)
            ax_ratio.set_ylabel("Target / Rw Original", fontsize=22)
            ax_ratio.set_ylim(0.50, 1.50)
            ax_ratio.legend(fontsize = 15)
            mh.set_fitting_ylabel_fontsize(ax_ratio)
            # plt.tight_layout()
            #plt.suptitle(f"Original, Reweighted and Target distributions with Ratio Plot (rw_dim = {len(list_parameters)})", y=1.02, fontsize=12)
            # mh.add_text(f"Original, Reweighted and Target distributions", ax = ax_main, loc = "over left", fontsize = 17)
            pdf.savefig(fig)
            plt.close(fig)

    return None

def plot_2D_histogram(original, target, weights_dict, target_weights = None, xlabels = ["E_nu(GeV)", "E_lepton(GeV)", "Cos Theta_l"], 
                      nbins = 30, pull = False, output_file = "/vols/dune/jmm224/t2knova/reweighting/saved_figures/histograms_and_ratios_2D.pdf"):
    mh.style.use("DUNE")
    combinations = np.array(np.meshgrid(range(len(xlabels)), range(len(xlabels)))).T.reshape(-1, 2)
    with PdfPages(f"{output_file}") as pdf:
        for combination in combinations:
            i, j = combination
            if i > j:
                percents = np.linspace(0, 100, nbins)
                bins_i = np.percentile(target[:, i], percents)
                bins_j = np.percentile(target[:, j], percents)
                if target_weights is not None:
                    H_target, x_edges, y_edges = np.histogram2d( target[:, i], target[:, j], bins=(bins_i, bins_j), weights=target_weights/np.sum(target_weights))
                else:
                    H_target, x_edges, y_edges = np.histogram2d( target[:, i], target[:, j], bins=(bins_i, bins_j), weights=np.ones_like(target[:, i])/len(target))
                for key in weights_dict.keys():
                    if pull:
                        H_rw, x_edges, y_edges = np.histogram2d( original[:, i], original[:, j], bins=(bins_i, bins_j), weights=weights_dict[key]/np.sum(weights_dict[key]))
                        H_rw_uncert = np.histogram2d( original[:, i], original[:, j], bins=(bins_i, bins_j), weights=(weights_dict[key]/np.sum(weights_dict[key]))**2)[0]
                        sigma = np.sqrt(H_rw_uncert)

                        H_pull = np.divide(
                            H_rw - H_target,
                            sigma,
                            out=np.zeros_like(H_rw),
                            where=sigma > 0
                        )

                        pull_masked = np.ma.masked_where(sigma == 0, H_pull)
                        # colormap with white for masked
                        cmap = plt.cm.coolwarm.copy()
                        cmap.set_bad(color='white')

                        # symmetric color scale around 0
                        max_delta = np.max(np.abs(pull_masked))
                        normalized_pull = pull_masked / max_delta
                        delta = np.max(np.abs(normalized_pull))
                        vmin, vmax = -delta, delta
                        fig, ax = plt.subplots(figsize=(8, 8))

                        im = ax.imshow(
                            normalized_pull.T,
                            origin='lower',
                            aspect='auto',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax
                        )

                        fig.colorbar(im, ax=ax, label='Statistical Pull (Normalized by max pull)')

                        ax.set_xlabel(f"{xlabels[i]}")
                        ax.set_ylabel(f"{xlabels[j]}")
                        ax.set_title(f"2D Histogram Pull ({key})")

                        pdf.savefig(fig)
                        plt.close(fig)

                    else:
                        H_rw, x_edges, y_edges = np.histogram2d( original[:, i], original[:, j], bins=(bins_i, bins_j), weights=weights_dict[key]/np.sum(weights_dict[key]))
                        ratio = np.divide(H_rw, H_target, out=np.zeros_like(H_rw), where=H_target > 0)

                        # mask invalid bins
                        ratio_masked = np.ma.masked_where(H_rw == 0, ratio)

                        # colormap with white for masked
                        cmap = plt.cm.coolwarm.copy()
                        cmap.set_bad(color='white')

                        # symmetric color scale around 1
                        delta = np.max(np.abs(ratio_masked - 1))
                        vmin, vmax = -delta, delta

                        fig, ax = plt.subplots(figsize=(8, 8))

                        im = ax.imshow(
                            ratio_masked.T,
                            origin='lower',
                            aspect='auto',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax
                        )

                        fig.colorbar(im, ax=ax, label='(Reweighted Original / Target) - 1 ')
                        ax.set_xlabel(f"{xlabels[i]}")
                        ax.set_ylabel(f"{xlabels[j]}")
                        ax.set_title(f"2D Histogram Ratio ({key})")

                        pdf.savefig(fig)
                        plt.close(fig)

    return None

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

def chi2_hist_naxis(original, target, rw_weights, binning_dict, target_weights = None, List_param_interest = ["Enu_true", "Plep", "CosLep"]):
    n = original.shape[1]
    chi2 = 0
    dof = 0
    for var in List_param_interest:
        axis_number = List_param_interest.index(var)
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

def chi2_dof(original, target, weights_dict, binning_dict, target_weights = None, List_param_interest = ["Enu_true", "Plep", "CosLep"]):
    dict_chi2 = {}
    for key in weights_dict.keys():
        chi2, dof = chi2_hist_naxis(original, target, weights_dict[key], binning_dict, target_weights, List_param_interest)
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
            W_dist = ot.wasserstein_1d(orig_proj, target_proj, 
                                       u_weights = weights_dict[key]/np.sum(weights_dict[key]), 
                                       v_weights=target_weights/np.sum(target_weights) if target_weights is not None else None)
            dict_swd[key].append(W_dist)
        
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
            W_dist = ot.wasserstein_1d(first_proj, second_proj, 
                                       u_weights = target_weights_bootstrap_1/np.sum(target_weights_bootstrap_1) if target_weights is not None else np.ones_like(target_bootstrap_1[:, 0])/len(target_bootstrap_1),
                                       v_weights = target_weights_bootstrap_2/np.sum(target_weights_bootstrap_2) if target_weights is not None else np.ones_like(target_bootstrap_2[:, 0])/len(target_bootstrap_2))

            list_wass.append(W_dist)
        
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
