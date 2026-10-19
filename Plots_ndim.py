"""Plots comparing the original, reweighted and target distributions.

The functions below take as input the original and target distributions and the weights predicted by
any reweighting method, and write the 1D histograms (with the ratio of the reweighted distribution
over the target one), the triangle plots (the original and target distributions, the target over the
reweighted distribution and the pull between them) and the training history of the models that carry
one. The metrics they display are computed by 'Metrics_ndim.py'.

The weights need not be normalised (ie enforcing np.sum(weights) = 1): the normalisation is included
in the plots themselves."""

#Imports

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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

def use_latex_default_font():
    """Switch matplotlib's font to LaTeX's default (Computer Modern), overriding the one 'mh.style.use'
    sets. This only changes matplotlib's own font choice, not 'text.usetex': it stays fast and does not
    need a LaTeX installation. The tick labels are written in mathtext, since cmr10 itself has no
    minus sign glyph."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True})

def plot_histograms(original, target, weights_dict, dict_binning, original_weights = None, target_weights = None,
                    xlabels = ["E_nu(GeV)", "E_lepton(GeV)", "cos_theta_lepton"],
                    variables = ["E_nu", "E_lepton", "cos_theta_lepton"],
                    add_wass_distance = True, add_chi2 = True, n_events = None,
                    output_file = "/vols/dune/jmm224/t2knova/reweighting/saved_figures/live.pdf"):
    """This function plots the original, reweighted and target distributions for each variable, with a ratio plot of the reweighted
      distribution over the target distribution. The weights_dict is a dictionary containing the predicted weights for each method,
      with the method name as key and the weights as value.

      'n_events' rescales the statistical uncertainties (and so the chi2/dof) as if 'original' and 'target' each
      held that many events instead of their actual number: as with 'chi2_hist_axis', it is None by default,
      which leaves the uncertainties at their actual (usually much smaller) statistical scale."""
    mh.style.use("DUNE")
    use_latex_default_font()
    target_norm = np.sum(target_weights) if target_weights is not None else len(target)
    # Each of the uncertainties below is a variance that scales as 1/n_events (see 'chi2_hist_axis'
    # for why this holds regardless of the events' weight magnitude): scaling it by the actual
    # number of events over a hypothetical smaller one reports the uncertainty that hypothetical
    # sample would give, rather than this (usually much larger) test sample's.
    target_scale = len(target) / n_events if n_events is not None else 1.0
    original_scale = len(original) / n_events if n_events is not None else 1.0
    with PdfPages(f"{output_file}") as pdf:
        for var in variables:
            #For the sake of the plots in this function, we use a given binning for each variable, specified in the dict_binning.
            #If the variable is not present, we use a default binning : uniform between the 1st and 99th percentiles of the 
            #target distribution for this variable.
            i = variables.index(var)
            if var in dict_binning.keys():
                x_min = dict_binning[var]["x_min"]
                x_max = dict_binning[var]["x_max"]
                n_bins = dict_binning[var]["n_bins"]
            else:
                first_percentile = np.percentile(target[:, i], 1)
                if np.abs(first_percentile) < 5e-2:
                    first_percentile = 0
                ninety_ninth_percentile = np.percentile(target[:, i], 99)
                x_min = first_percentile
                x_max = ninety_ninth_percentile
                n_bins = 30
            bins = np.linspace(x_min, x_max, n_bins+1)

            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Getting the distributions and their uncertainties, with a normalization to the total number of events.
            # The original and target distribution are normalized to 1. The reweighted distributions are normalized by the same 
            # factor as the target distribution.

            if target_weights is not None:
                target_counts = np.histogram(target[:, i], bins=bins, weights = target_weights/target_norm)[0]
                target_counts_uncert = np.histogram(target[:, i], bins=bins, weights= (target_weights/target_norm)**2)[0]
            else:
                target_counts = np.histogram(target[:, i], bins=bins, weights = np.ones_like(target[:, i])/len(target))[0]
                target_counts_uncert = np.histogram(target[:, i], bins=bins, weights= (np.ones_like(target[:, i])/len(target))**2)[0]

            if original_weights is not None:
                original_counts = np.histogram(original[:, i], bins=bins, weights = original_weights/np.sum(original_weights))[0]
                orig_counts_uncert = np.histogram(original[:, i], bins=bins, weights= (original_weights/np.sum(original_weights))**2)[0]
            else:
                original_counts = np.histogram(original[:, i], bins=bins, weights = np.ones_like(original[:, i])/len(original))[0]
                orig_counts_uncert = np.histogram(original[:, i], bins=bins, weights= (np.ones_like(original[:, i])/len(original))**2)[0]

            target_uncertainty = np.sqrt(target_counts_uncert * target_scale)
            orig_uncertainty = np.sqrt(orig_counts_uncert)

            # Getting the distribution for the reweighted original distributions, and its uncertainty (propagated from the weights uncertainty)
        
            original_rw_counts_list = []
            original_rw_counts_uncert_list = []
        
            for k in range(len(weights_dict)):
                key = list(weights_dict.keys())[k]
                original_rw_counts, _ = np.histogram(original[:, i], bins=bins, weights= weights_dict[key]/target_norm)
                original_rw_counts_uncert = np.histogram(original[:, i], bins=bins, weights= (weights_dict[key]/target_norm)**2)[0]
                original_rw_uncertainty = np.sqrt(original_rw_counts_uncert * original_scale)
                original_rw_counts_list.append(original_rw_counts)
                original_rw_counts_uncert_list.append(original_rw_uncertainty)


            fig, (ax_main, ax_ratio) = plt.subplots(
                2, 1, figsize=(8, 8), gridspec_kw={'height_ratios':[2,1], 'hspace': 0.1}, sharex=True,
                constrained_layout=True
            )

            colors = {}
            markers_list = ['s', 'd', '^', 'p', '*', 'h']  # Add more markers if needed
            chi2_values = {}
            norm_ratio_values = {}

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
                    chi2, dof = chi2_hist_axis(original, target, weights_dict[key], i, target_weights, n_bins=n_bins, x_min=x_min, x_max=x_max, n_events=n_events)
                    chi2_values[key] = chi2/dof if dof > 0 else 0
                    norm_ratio_values[key] = np.sum(weights_dict[key]) / target_norm

            # Distribution plot
            ax_main.errorbar(bin_centers, target_counts, yerr=target_uncertainty, fmt='o', markersize=4, label='Target')
            ax_main.set_xlim(bins[0], bins[-1])
        
            for k in range(len(weights_dict)):
                key = list(weights_dict.keys())[k]
                line_rw, = ax_main.step(bins, np.r_[original_rw_counts_list[k],
                                        original_rw_counts_list[k][-1]],
                                        markersize=0,
                                        where='post',
                                        label=(f'{key[0].upper() + key[1:]}' + r" $\chi^2_{dof}$:" + f" {chi2_values[key]:.2f}"
                                               + f", N: {norm_ratio_values[key]:.2f}") if add_chi2 else f'{key[0].upper() + key[1:]}')
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
            ax_main.legend(fontsize = 12, ncols = 3, loc = 'upper center', handlelength=1.2)
        

            # Ratio plot (Data / Reweighted Original)

            for k in range(len(weights_dict)):
                ratio = target_counts / original_rw_counts_list[k]

                # Reweighting uncertainty for the points (propagated to ratio)
                ratio_rw_err = ratio * (original_rw_counts_uncert_list[k] / original_rw_counts_list[k])

                # Plot ratio points with reweighting uncertainty
                ax_ratio.step(bins, np.r_[ratio, ratio[-1]],
                              where = 'post', 
                              markersize = 0.1, 
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
            ax_ratio.legend(fontsize = 12)
            mh.set_fitting_ylabel_fontsize(ax_ratio)
            pdf.savefig(fig)
            plt.close(fig)

        # Plot weight distribution for each method
        if original_weights is not None:
            # The weight each method predicted on its own, without the pre-weight of the events.
            # 'Original' isn't a method's prediction, so dividing it by itself would only give a
            # trivial spike at 1.
            bdt_weights_dict = {key: weights / original_weights for key, weights in weights_dict.items()
                                if key != "Original"}

            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

            # Bins shared by every method, so their distributions are directly comparable. The upper
            # edge is a percentile rather than the max, so that the handful of largest weights a
            # density-ratio method can produce don't crush every other bin flat.
            all_weights = np.concatenate(list(bdt_weights_dict.values()))
            bins = np.linspace(np.min(all_weights), np.percentile(all_weights, 99.5), 51)

            for key, weights in bdt_weights_dict.items():
                ax.hist(weights, bins=bins, histtype='step', linewidth=1.5,
                       color=colors.get(key), label=key[0].upper() + key[1:])

            ax.set_xlabel("BDT weight", fontsize=22)
            ax.set_ylabel("Events", fontsize=22)
            ax.set_yscale("log")
            ax.legend(fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

    return None

# Chart colours: a sequential blue ramp for magnitudes, and a blue <-> red ramp diverging about a
# neutral grey for quantities with a meaningful midpoint, on a near-white surface.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SEQUENTIAL_COLOURS = [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_COLOURS = ["#8a2323", "#e34948", "#f0efec", "#2a78d6", "#104281"]
LINE_COLOUR = "#2a78d6"
UNDEFINED_HATCH = "#a8a69f"

def triangle_edges(target, params, binning_dict, n_bins):
    """Return the bin edges of every parameter of a triangle plot: 'n_bins' uniform bins over the range
    of the binning file when it lists the parameter, and over the 1-99% range of the target otherwise
    (as 'plot_histograms' does)."""
    edges = []
    for i, param in enumerate(params):
        if param in binning_dict:
            x_min, x_max = binning_dict[param]["x_min"], binning_dict[param]["x_max"]
        else:
            x_min, x_max = np.percentile(target[:, i], [1, 99])
            if np.abs(x_min) < 5e-2:
                x_min = 0
        edges.append(np.linspace(x_min, x_max, n_bins + 1))
    return edges

def _histogram(sample, weights, i, j, edges):
    """Return the weighted histogram of a sample in parameter i (j None) or in (i, j)."""
    if j is None:
        return np.histogram(sample[:, i], bins=edges[i], weights=weights)[0]
    return np.histogram2d(sample[:, i], sample[:, j], bins=(edges[i], edges[j]), weights=weights)[0]

def _draw_triangle(pdf, edges, labels, cell, diagonal, cmap, norm, colour_label, diagonal_label, title,
                   diagonal_ylim=None, log_ticks=None, reference_lines=(), extend="neither", hatch_undefined=False):
    """Draw one page of a triangle plot. The panel in row r and column c < r shows cell(c, r) (a 2D
    array over the bins of parameters c and r, masked where undefined), and the diagonal panel c
    shows diagonal(c) (an array over the bins of parameter c, NaN where undefined).

    'log_ticks' puts the diagonal panels and the colour bar on a log scale, labelled at those values.
    'hatch_undefined' hatches the masked cells, so that they read apart from every colour of the scale
    (the ratio's near-white midpoint above all); it also leaves the diagonal outlines open, as a value
    of 0 would have no meaning there."""
    n = len(edges)
    fig, axes = plt.subplots(n, n, figsize=(2.3 * n, 2.3 * n), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.08, hspace=0.08, left=0.08, right=0.97, bottom=0.07, top=0.97)
    tick_labels = [f"{tick:g}" for tick in log_ticks] if log_ticks is not None else None
    mappable = None
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            if c > r:
                ax.set_axis_off()
                continue
            if c == r:
                ax.stairs(diagonal(c), edges[c], color=LINE_COLOUR, linewidth=2,
                          baseline=None if hatch_undefined else 0)
                for y in reference_lines:
                    ax.axhline(y, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
                if log_ticks is not None:
                    ax.set_yscale("log")
                    ax.set_yticks(log_ticks, tick_labels)
                    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if diagonal_ylim is not None:
                    ax.set_ylim(*diagonal_ylim)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                if c == 0:
                    ax.set_ylabel(diagonal_label, fontsize=8, color=TEXT_SECONDARY)
            else:
                if hatch_undefined:
                    ax.patch.set_hatch("////")
                    ax.patch.set_edgecolor(UNDEFINED_HATCH)
                    ax.patch.set_linewidth(0)
                mappable = ax.pcolormesh(edges[c], edges[r], cell(c, r).T, cmap=cmap, norm=norm,
                                         rasterized=True)
                ax.set_ylim(edges[r][0], edges[r][-1])
                # Pruned, so that the end labels of neighbouring panels do not run into each other.
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune="both"))
            ax.set_xlim(edges[c][0], edges[c][-1])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune="both"))
            ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
            for spine in ax.spines.values():
                spine.set_color(TEXT_SECONDARY)
                spine.set_linewidth(0.5)
            if r == n - 1:
                ax.set_xlabel(labels[c], fontsize=9, color=TEXT_PRIMARY)
            else:
                ax.tick_params(labelbottom=False)
            if c == 0 and r > 0:
                ax.set_ylabel(labels[r], fontsize=9, color=TEXT_PRIMARY)
            elif c != r:
                ax.tick_params(labelleft=False)
    # The title and the colour bar sit in the empty upper triangle.
    fig.text(0.97, 0.95, title, ha="right", va="top", fontsize=12, color=TEXT_PRIMARY)
    if mappable is not None:
        cax = fig.add_axes([0.6, 0.86, 0.35, 0.018])
        bar = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend)
        if log_ticks is not None:
            bar.set_ticks(log_ticks, labels=tick_labels)
            bar.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        bar.set_label(colour_label, fontsize=9, color=TEXT_SECONDARY)
        bar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
        bar.outline.set_visible(False)
    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)

def plot_triangles(original, target, weights_dict, params, binning_dict, original_weights, target_weights,
                   output_dir, labels=None, title_suffix="", n_bins=20, ratio_range=(0.5, 2.0), max_pull=5.0,
                   density_floor=1e-3, pull_target=None, pull_title_suffix=None):
    """Write the four triangle plots of a set of parameters: every parameter on the diagonal and every
    pair of them below it.

      - 'triangle_original.pdf' and 'triangle_target.pdf': the original and target distributions,
        normalised to 1. The 2D panels are shown relative to the peak of the same panel over both
        samples, so the two files are on one colour scale, and on a log scale down to 'density_floor'
        of that peak, so that the tails of the distributions remain visible beside their peak.
      - 'triangle_ratio.pdf': target / reweighted, one page per entry of 'weights_dict' (the absolute
        weights of a model, or the original pre-weights for 'Original'). Both are normalised to the
        target's total, as in 'plot_histograms', so a normalisation mismatch shows as an offset.
      - 'triangle_pull.pdf': |reweighted - target| / sigma_target, sigma_target being the target's
        statistical uncertainty in the bin, one page per entry of 'weights_dict'. The reweighted
        original is compared to 'pull_target', a (target, target_weights) tuple (ex: a data-sized
        target subsample, so that sigma_target is the uncertainty of the data, as given by
        'Metrics_ndim.data_sized_target'), and to the target above by default; normalised as
        'triangle_ratio.pdf' is, and titled with 'pull_title_suffix' ('title_suffix' by default). Only
        the target's uncertainty is in the denominator, so the original sample should be much larger
        than 'pull_target' for the pull of a perfect reweighting to spread as a unit one would.

    Every plot is binned alike, from the target sample above. Bins where a quantity is undefined (no
    target events, or nothing reweighted) are hatched."""
    labels = labels if labels is not None else labels_of(params)
    edges = triangle_edges(target, params, binning_dict, n_bins)
    target_norm = np.sum(target_weights)
    original_norm = np.sum(original_weights)
    sequential = matplotlib.colors.LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_COLOURS)
    diverging = matplotlib.colors.LinearSegmentedColormap.from_list("diverging", DIVERGING_COLOURS)
    pairs = [(c, r) for r in range(len(params)) for c in range(r)]

    with plt.rc_context():
        plt.style.use("default")
        use_latex_default_font()

        # Original and target, on one colour scale per 2D panel and one y range per diagonal panel.
        density = {"original": {}, "target": {}}
        for name, sample, weights, norm in (("original", original, original_weights, original_norm),
                                            ("target", target, target_weights, target_norm)):
            for c, r in pairs:
                density[name][(c, r)] = _histogram(sample, weights / norm, c, r, edges)
            for c in range(len(params)):
                density[name][c] = _histogram(sample, weights / norm, c, None, edges)
        peak = {key: max(density["original"][key].max(), density["target"][key].max()) for key in density["target"]}
        for name, title in (("original", "Original"), ("target", "Target")):
            with PdfPages(os.path.join(output_dir, f"triangle_{name}.pdf")) as pdf:
                _draw_triangle(pdf, edges, labels,
                               cell=lambda c, r: density[name][(c, r)] / peak[(c, r)] if peak[(c, r)] > 0 else density[name][(c, r)],
                               diagonal=lambda c: density[name][c],
                               cmap=sequential, norm=matplotlib.colors.LogNorm(density_floor, 1),
                               colour_label="density relative to the panel's peak (original and target)",
                               diagonal_label="fraction of events",
                               diagonal_ylim=(0, 1.1 * max(peak[c] for c in range(len(params)))),
                               extend="min", title=f"{title} distributions{title_suffix}")

        # Target over reweighted, and the pull of reweighted from target, for every set of weights.
        # Both the target and reweighted distributions are normalised to the target's total.
        def target_and_reweighted(original, target, weights, target_weights, c, r):
            norm = np.sum(target_weights)
            t = _histogram(target, target_weights / norm, c, r, edges)
            t_variance = _histogram(target, (target_weights / norm) ** 2, c, r, edges)
            rw = _histogram(original, weights / norm, c, r, edges)
            return t, np.sqrt(t_variance), rw

        def ratio(weights, c, r=None):
            t, _, rw = target_and_reweighted(original, target, weights, target_weights, c, r)
            return np.ma.masked_where((t <= 0) | (rw <= 0), np.divide(t, rw, out=np.ones_like(t), where=rw > 0))

        pull_target, pull_target_weights = pull_target if pull_target is not None else (target, target_weights)
        pull_title_suffix = pull_title_suffix if pull_title_suffix is not None else title_suffix

        def pull(weights, c, r=None):
            t, sigma, rw = target_and_reweighted(original, pull_target, weights, pull_target_weights, c, r)
            return np.ma.masked_where(sigma <= 0, np.divide(np.abs(rw - t), sigma, out=np.zeros_like(t), where=sigma > 0))

        # The ratio is drawn on a log scale, symmetric about 1, so that a ratio and its inverse are
        # equally far from the neutral midpoint.
        ratio_ticks = [ratio_range[0], 1, ratio_range[1]]
        with PdfPages(os.path.join(output_dir, "triangle_ratio.pdf")) as pdf:
            for key, weights in weights_dict.items():
                _draw_triangle(pdf, edges, labels,
                               cell=lambda c, r: ratio(weights, c, r), diagonal=lambda c: ratio(weights, c).filled(np.nan),
                               cmap=diverging, norm=matplotlib.colors.LogNorm(*ratio_range),
                               colour_label="target / reweighted", diagonal_label="target / reweighted",
                               diagonal_ylim=ratio_range, log_ticks=ratio_ticks, reference_lines=(1,), extend="both",
                               hatch_undefined=True, title=f"Target / {key}{title_suffix}")
        with PdfPages(os.path.join(output_dir, "triangle_pull.pdf")) as pdf:
            for key, weights in weights_dict.items():
                _draw_triangle(pdf, edges, labels,
                               cell=lambda c, r: pull(weights, c, r), diagonal=lambda c: pull(weights, c).filled(np.nan),
                               cmap=sequential, norm=matplotlib.colors.Normalize(0, max_pull),
                               colour_label=r"$|$reweighted $-$ target$|$ / $\sigma_{target}$",
                               diagonal_label=r"$|$reweighted $-$ target$|$ / $\sigma_{target}$",
                               diagonal_ylim=(0, max_pull), reference_lines=(1, 3), extend="max",
                               hatch_undefined=True, title=f"Pull of {key} from target{pull_title_suffix}")

def plot_training_history(model, output_file, title="Training history"):
    """Plot the training and validation log loss of a model against the boosting iteration.

    Only the models carrying an 'evals_result' method (the XGBoost ones) have such a history:
    nothing is written for the others, and False is returned."""
    if getattr(model, "evals_result_", None) is None:
        print(f"Warning: the model has no training history: {output_file} is not written.")
        return False

    use_latex_default_font()
    training_history = model.evals_result()
    with PdfPages(output_file) as pdf:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(training_history["validation_0"]["logloss"], label='Train Log Loss')
        ax.plot(training_history["validation_1"]["logloss"], label='Validation Log Loss')
        ax.set_xlabel('Boosting Iteration')
        ax.set_ylabel('Log Loss')
        ax.set_title(title)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
    return True
