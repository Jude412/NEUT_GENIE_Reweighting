"""Plots comparing the original, reweighted and target distributions.

The functions below take as input the original and target distributions and the weights predicted by
any reweighting method, and write the 1D histograms (with the ratio of the reweighted distribution
over the target one), the 2D histograms (ratio or statistical pull) and the training history of the
models that carry one. The metrics they display are computed by 'Metrics_ndim.py'.

The weights need not be normalised (ie enforcing np.sum(weights) = 1): the normalisation is included
in the plots themselves."""

#Imports

import numpy as np
import matplotlib.pyplot as plt
import ot
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as mh

from Metrics_ndim import chi2_hist_axis

# Labels of the analysis parameters, used to axis-label the plots. Parameters that are not listed
# are labelled with their own name.
PARAMETER_LABELS = {
    "Enu_true": r"$E^{true}_{\nu}$ (GeV)",
    "Plep": r"$p_{lep}$ (GeV/c)",
    "PLep": r"$p_{lep}$ (GeV/c)",
    "CosLep": r"$cos(\theta_{lep})$",
    "Q2": r"$Q^2$ (GeV$^2$/c$^2$)",
    "q0": r"$q_0$ (GeV)",
    "q3": r"$q_3$ (GeV/c)",
    "PTlep": r"$p^T_{lep}$ (GeV/c)",
    "Eav": r"$E_{Av}$ (GeV)",
    "W": r"$W$ (GeV/c$^2$)",
    "y": "y",
    "Mode": "Mode",
    "Topology": "Topology",
    "cc": "cc",
    "hitnuc": "hitnuc",
    "N_n": r"$N_n$",
    "K_n": r"$K_n$ (GeV)",
    "N_p": r"$N_p$",
    "K_p": r"$K_p$ (GeV)",
    "N_pi0": r"$N_{pi^0}$",
    "K_pi0": r"$K_{pi^0}$ (GeV)",
    "N_pip": r"$N_{pi^+}$",
    "K_pip": r"$K_{pi^+}$ (GeV)",
    "N_pim": r"$N_{pi^-}$",
    "K_pim": r"$K_{pi^-}$ (GeV)"
}

def labels_of(params):
    """Return the labels of the given analysis parameters, defaulting to their own name."""
    return [PARAMETER_LABELS.get(param, param) for param in params]

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
                bins = np.linspace(x_min, x_max, n_bins+1)
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
                    chi2, dof = chi2_hist_axis(original, target, weights_dict[key], i, target_weights, n_bins=n_bins, x_min=x_min, x_max=x_max)
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
                    hist_target, x_edges, y_edges = np.histogram2d( target[:, i], target[:, j], bins=(bins_i, bins_j), weights=target_weights/np.sum(target_weights))
                else:
                    hist_target, x_edges, y_edges = np.histogram2d( target[:, i], target[:, j], bins=(bins_i, bins_j), weights=np.ones_like(target[:, i])/len(target))
                for key in weights_dict.keys():
                    if pull:
                        hist_rw, x_edges, y_edges = np.histogram2d( original[:, i], original[:, j], bins=(bins_i, bins_j), weights=weights_dict[key]/np.sum(weights_dict[key]))
                        hist_rw_uncert = np.histogram2d( original[:, i], original[:, j], bins=(bins_i, bins_j), weights=(weights_dict[key]/np.sum(weights_dict[key]))**2)[0]
                        sigma = np.sqrt(hist_rw_uncert)

                        hist_pull = np.divide(
                            hist_rw - hist_target,
                            sigma,
                            out=np.zeros_like(hist_rw),
                            where=sigma > 0
                        )

                        pull_masked = np.ma.masked_where(sigma == 0, hist_pull)
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
                        hist_rw, x_edges, y_edges = np.histogram2d( original[:, i], original[:, j], bins=(bins_i, bins_j), weights=weights_dict[key]/np.sum(weights_dict[key]))
                        ratio = np.divide(hist_rw, hist_target, out=np.zeros_like(hist_rw), where=hist_target > 0)

                        # mask invalid bins
                        ratio_masked = np.ma.masked_where(hist_rw == 0, ratio)

                        # colormap with white for masked
                        cmap = plt.cm.coolwarm.copy()
                        cmap.set_bad(color='white')

                        # symmetric color scale around 1
                        ratio_delta = ratio_masked - 1
                        delta = np.max(np.abs(ratio_delta))
                        vmin, vmax = -delta, delta

                        fig, ax = plt.subplots(figsize=(8, 8))

                        im = ax.imshow(
                            ratio_delta.T,
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

def plot_training_history(model, output_file, title="Training history"):
    """Plot the training and validation log loss of a model against the boosting iteration.

    Only the models carrying an 'evals_result' method (the XGBoost ones) have such a history:
    nothing is written for the others, and False is returned."""
    if not hasattr(model, "evals_result"):
        print(f"Warning: the model has no training history: {output_file} is not written.")
        return False

    training_history = model.evals_result()
    with PdfPages(output_file) as pdf:
        fig, ax = plt.subplots()
        ax.plot(training_history["validation_0"]["logloss"], label='Train Log Loss')
        ax.plot(training_history["validation_1"]["logloss"], label='Validation Log Loss')
        ax.set_xlabel('Boosting Iteration')
        ax.set_ylabel('Log Loss')
        ax.set_title(title)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
    return True
