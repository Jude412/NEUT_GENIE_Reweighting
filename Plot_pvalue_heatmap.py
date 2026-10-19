"""Plot heatmaps of the data-scale SWD p-values of every sample and topology of a tag.

One figure is drawn per set of parameters, with one panel per top-level sample group (ex: the
crpa, dcc and nominal_multipi variants), the samples within it as rows and the topologies as
columns. Each cell is the data-scale p-value of the selected model of that sample and topology, as
written to its 'metrics.json' by 'Gather_metrics.py', or of the un-reweighted original sample.

The colour scale diverges at the pass threshold (0.05 by default) on a log axis: red fails, blue
passes, and the neutral grey midpoint sits on the threshold itself. The p-values come from a finite
bootstrapped SWD distribution, so a p-value of 0 only means smaller than the distribution can
resolve: those cells are drawn at the floor of the scale ('--p_floor')."""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

# Diverging blue <-> red with a neutral grey midpoint, from the reference chart palette.
FAIL_COLOURS = ["#8a2323", "#e34948"]
MID_COLOUR = "#f0efec"
PASS_COLOURS = ["#2a78d6", "#104281"]
SURFACE = "#fcfcfb"
MISSING_HATCH = "#a8a69f"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

FLAVOUR_LABELS = {"nue": r"$\nu_e$", "nueb": r"$\bar{\nu}_e$", "numu": r"$\nu_\mu$", "numub": r"$\bar{\nu}_\mu$"}
TOPOLOGY_ORDER = ["CC0pi", "CC1pipm", "CC1pi0", "CCNpi", "CCOther",
                  "NC0pi", "NC1pipm", "NC1pi0", "NCNpi", "NCOther", "Other"]

def read_p_values(metrics_dir, model, param_set, metrics_files=None):
    """Return the {(group, sample, topology): p-value} dictionary of the metrics.json files of a tag:
    'metrics_files' if given, and every one under 'metrics_dir' otherwise.

    'group' is the first directory of the sample path relative to 'metrics_dir' (the panel it is
    drawn in) and 'sample' the rest of it. 'model' is a model of the metrics files, or 'Original' for
    the un-reweighted sample, whose baseline p-value every model's metrics carry."""
    if metrics_files is None:
        metrics_files = glob.glob(os.path.join(metrics_dir, "**", "metrics.json"), recursive=True)
    p_values = {}
    for f in metrics_files:
        path = os.path.relpath(os.path.dirname(f), metrics_dir).split(os.sep)
        group, sample, topology = path[0], "/".join(path[1:-1]), path[-1]
        with open(f) as fh:
            metrics = json.loads(fh.read().replace("NaN", "null"))
        if model == "Original":
            value = next(iter(metrics.values()))["metrics"].get(f"Original_data_p_value_{param_set}")
        else:
            if model not in metrics:
                continue
            value = metrics[model]["metrics"].get(f"data_p_value_{param_set}")
        if value is not None:
            p_values[(group, sample, topology)] = value
    return p_values

def sample_label(sample):
    """Return the row label of a sample path, ex 'ND280/CH2_C/numu' -> 'ND280 CH2_C ν_μ'."""
    parts = sample.split("/")
    return " ".join(FLAVOUR_LABELS.get(part, part) for part in parts)

def plot_heatmap(p_values, param_set, model, threshold, p_floor, output_file):
    groups = sorted({key[0] for key in p_values})
    samples = sorted({key[1] for key in p_values})
    present = {key[2] for key in p_values}
    topologies = [t for t in TOPOLOGY_ORDER if t in present] + sorted(present - set(TOPOLOGY_ORDER))

    cmap = LinearSegmentedColormap.from_list("p_value", FAIL_COLOURS + [MID_COLOUR] + PASS_COLOURS)
    cmap.set_bad(SURFACE)
    norm = TwoSlopeNorm(vmin=np.log10(p_floor), vcenter=np.log10(threshold), vmax=0)

    fig, axes = plt.subplots(1, len(groups), figsize=(1.0 + 3.6 * len(groups), 0.9 + 0.24 * len(samples)),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for ax, group in zip(axes[0], groups):
        grid = np.full((len(samples), len(topologies)), np.nan)
        for i, sample in enumerate(samples):
            for j, topology in enumerate(topologies):
                p = p_values.get((group, sample, topology))
                if p is not None:
                    grid[i, j] = np.log10(max(p, p_floor))
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto",
                          interpolation="nearest")
        # Topologies not included for a sample are hatched rather than filled, so that no fill
        # colour of the scale (the grey midpoint above all) can be mistaken for them.
        for i, j in zip(*np.where(np.isnan(grid))):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SURFACE, edgecolor=MISSING_HATCH,
                                   hatch="////", linewidth=0))
        # A 1-pixel surface gap between cells, so neighbouring cells of close colour stay separable.
        ax.set_xticks(np.arange(-0.5, len(topologies)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(samples)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(len(topologies)), topologies, rotation=60, ha="right", fontsize=8,
                      color=TEXT_SECONDARY)
        ax.set_yticks(range(len(samples)), [sample_label(s) for s in samples], fontsize=8,
                      color=TEXT_SECONDARY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        n_pass = np.sum(grid >= np.log10(threshold)); n_total = np.sum(~np.isnan(grid))
        ax.set_title(f"{group}\n{n_pass}/{n_total} pass", fontsize=9, color=TEXT_PRIMARY)

    ticks = [p for p in (p_floor, 0.01, threshold, 0.2, 1.0) if p >= p_floor]
    bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02, ticks=np.log10(ticks))
    bar.ax.set_yticklabels([f"<{p_floor:g}" if p == p_floor else f"{p:g}" for p in ticks],
                           fontsize=8, color=TEXT_SECONDARY)
    bar.set_label(f"data-scale p-value (fails below {threshold:g})", fontsize=8, color=TEXT_SECONDARY)
    bar.outline.set_visible(False)
    model_label = "before reweighting" if model == "Original" else model
    fig.suptitle(f"SWD p-value on the {param_set} parameters, {model_label} "
                 f"(hatched: topology not included)", fontsize=10, color=TEXT_PRIMARY)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Plot heatmaps of the data-scale SWD p-values of every sample and topology of a tag.")
    argparser.add_argument("--metrics_dir", type=str, required=True,
                           help="Directory of the metrics of a tag, ex: saved_metrics/complete_0/.")
    argparser.add_argument("--metrics_files", nargs="+", default=None,
                           help="The metrics.json files under '--metrics_dir' to plot. Every one under it "
                                "by default.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory the figures are saved in.")
    argparser.add_argument("--models", nargs="+", default=["XGB"],
                           help="Models to plot the selected p-values of, or 'Original' for the un-reweighted sample.")
    argparser.add_argument("--param_sets", nargs="+", default=["reweighting", "NIWG", "Sel"],
                           help="Sets of parameters to plot the p-values of, one figure each.")
    argparser.add_argument("--threshold", type=float, default=0.05, help="p-value a sample and topology passes at.")
    argparser.add_argument("--p_floor", type=float, default=1e-3,
                           help="Smallest p-value drawn: the resolution of the bootstrapped SWD distribution "
                                "(1 / its number of samples).")
    args = argparser.parse_args()

    for model in args.models:
        for param_set in args.param_sets:
            p_values = read_p_values(args.metrics_dir, model, param_set, args.metrics_files)
            if not p_values:
                print(f"No p-values found for the model '{model}' and the set '{param_set}'.")
                continue
            plot_heatmap(p_values, param_set, model, args.threshold, args.p_floor,
                         os.path.join(args.output_dir, f"pvalue_heatmap_{model}_{param_set}.pdf"))
