"""Plot a metric of the fine-tuning runs against one or two of the hyperparameters they were run with.

'Compute_metrics.py' writes one 'metrics.csv' per point of the hyperparameter grid of a model, holding
that point's hyperparameters followed by every metric it was evaluated on, and 'Gather_metrics.py' then
keeps only the best row of the whole scan. This script reads those per-run files back, so that the shape
of the scan itself can be looked at: a metric against one hyperparameter (one line per combination of the
hyperparameters that are not plotted against) or against two (an annotated heatmap), for one model at a
time, cut down to a chosen tag, sample and topology.

The runs are read from 'saved_metrics/{tag}/{sample}/{topology}/{model}/{hp_id}/metrics.csv', where
'hp_id' is the label a point of the grid is named with by its hyperparameter values. A run that failed
simply has no file and leaves a gap, which is drawn as one rather than interpolated over. Only the tags
written since the metrics were split per run hold those files: the older ones hold a single
'metrics.json' per topology, in a different format, which this script does not read. The tags written
before the runs were labelled by their values hold numeric directories instead, which are read back the
same way: the label is only ever used as the name of the run."""

# imports
from List_hyperparameters import grid_of
from Plots_ndim import PARAMETER_LABELS, use_latex_default_font
from matplotlib.colors import LogNorm, Normalize
import argparse
import difflib
import fnmatch
import glob
import json
import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import os
import pandas as pd
import textwrap

# The fields identifying a run, read from the path of its metrics file and added to its columns.
RUN_COLUMNS = ("tag", "sample", "topology", "model", "hp_id")

# The fields a run can be cut on, and so the ones that get one panel each when several are kept.
CUT_COLUMNS = ("tag", "sample", "topology")

# The families the metric names of 'Compute_metrics.py' are built from, and the label they are given on
# the plots. The longest family matching a name wins, so that 'chi2_p_value_8D' is read as a chi2 p-value
# rather than as a p-value of the parameter set 'chi2'. 'p_value' is only written by runs from before the
# p-values were computed at the data scale, and is kept so that their metrics can still be plotted.
METRIC_FAMILIES = {
    "SWD": "SWD",
    "p_value": "p-value",
    "chi2_dof": r"$\chi^2$/dof",
    "chi2_p_value": r"$\chi^2$ p-value",
    "norm": r"$\Sigma w_{reweighted} / \Sigma w_{target}$",
    "data_SWD": "Data-scale SWD",
    "data_p_value": "Data-scale p-value",
    "data_chi2_dof": r"Data-scale $\chi^2$/dof",
    "data_chi2_p_value": r"Data-scale $\chi^2$ p-value",
}

# The metrics that carry no parameter set in their name, written for the XGBoost models only.
TRAINING_METRICS = ("val_logloss", "val_auc", "train_logloss", "train_auc")

# The prefix the un-reweighted baseline of a metric is named with, i.e. the same quantity taken between
# the original and the target samples without any reweighting.
BASELINE_PREFIX = "Original_"


def is_metric_name(name):
    """Whether a column of a metrics file holds a metric rather than a hyperparameter.

    The metric names are built by 'Compute_metrics.py' from a small set of families, so they can be told
    apart from the hyperparameters, whose names come from the grid file and are free-form."""
    name = name[len(BASELINE_PREFIX):] if name.startswith(BASELINE_PREFIX) else name
    return name in TRAINING_METRICS or any(name.startswith(f"{family}_") for family in METRIC_FAMILIES)


def metric_label(metric):
    """Return the axis label of a metric, ex 'chi2_dof_Enu_true' -> '$\\chi^2$/dof ($E^{true}_{\\nu}$)'.

    The parameter set or analysis parameter a metric is taken on is labelled with 'Plots_ndim' labels
    when it is one of the analysis parameters, and with its own name otherwise (ex: '8D')."""
    if metric.startswith(BASELINE_PREFIX):
        prefix, name = "Original ", metric[len(BASELINE_PREFIX):]
    else:
        prefix, name = "", metric
    for family in sorted(METRIC_FAMILIES, key=len, reverse=True):
        if name.startswith(f"{family}_"):
            suffix = name[len(family) + 1:]
            return f"{prefix}{METRIC_FAMILIES[family]} ({PARAMETER_LABELS.get(suffix, suffix)})"
    return prefix + name


def tags_with_metrics(metrics_dir):
    """Return the output tags holding per-run metrics csv files, i.e. the ones this script can read."""
    files = glob.glob(os.path.join(metrics_dir, "*", "**", "metrics.csv"), recursive=True)
    return sorted({os.path.relpath(path, metrics_dir).split(os.sep)[0] for path in files})


def run_of_path(path, metrics_dir):
    """Return the tag, sample, topology, model and hp_id a metrics file was written for.

    The sample is a directory path relative to the tag and may hold several segments ('ND280/CH/numu') or
    a single one ('test'), so the depth of the tree is not fixed and the fields are read from the right:
    the hp_id, the model and the topology are the last three directories, the tag is the first one, and
    the sample is everything in between."""
    parts = os.path.relpath(path, metrics_dir).split(os.sep)[:-1]
    if len(parts) < 5:
        raise ValueError(f"Cannot read the tag, sample, topology, model and hp_id of '{path}': the "
                         f"metrics files are expected in "
                         f"'{metrics_dir}/{{tag}}/{{sample}}/{{topology}}/{{model}}/{{hp_id}}/'.")
    return {"tag": parts[0], "sample": "/".join(parts[1:-3]),
            "topology": parts[-3], "model": parts[-2], "hp_id": parts[-1]}


def matches(value, patterns):
    """Whether a value is kept by a cut, given as a list of shell-style patterns (None keeping every one).

    The patterns are matched with 'fnmatch', whose '*' spans '/' as well, so that '--sample ND280/*'
    keeps every sample under 'ND280' whatever the depth of its path."""
    return patterns is None or any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def load_runs(metrics_dir, model, tags=None, samples=None, topologies=None):
    """Return one row per fine-tuning run of a model, holding its hyperparameters and its metrics.

    The columns are the fields identifying the run ('tag', 'sample', 'topology', 'model', 'hp_id')
    followed by every column of its metrics file. Runs that failed have no metrics file and so no row:
    the grid is always read back as the scatter of the points that did complete."""
    paths = sorted(glob.glob(os.path.join(metrics_dir, "*", "**", model, "*", "metrics.csv"),
                             recursive=True))
    rows = []
    for path in paths:
        run = run_of_path(path, metrics_dir)
        if (matches(run["tag"], tags) and matches(run["sample"], samples)
                and matches(run["topology"], topologies)):
            rows.append(pd.read_csv(path).assign(**run))

    if not rows:
        available = tags_with_metrics(metrics_dir)
        if not any(matches(tag, tags) for tag in available):
            raise ValueError(f"No per-run metrics file found under '{metrics_dir}' for the tag(s) "
                             f"{tags if tags is not None else 'given'}. The tags holding per-run metrics "
                             f"csv files are {available}: the other tags hold only the single "
                             f"'metrics.json' per topology written before the metrics were split per "
                             f"run, which this script does not read.")
        raise ValueError(f"No run of the model '{model}' matches the cuts "
                         f"(tag {tags}, sample {samples}, topology {topologies}). "
                         f"{available_pairs_message(metrics_dir, model, tags)}")

    runs = pd.concat(rows, ignore_index=True, sort=False)
    incomplete = [column for column in runs.columns if runs[column].isna().any()]
    if incomplete:
        print(f"Warning: the column(s) {incomplete} are missing from some of the runs read.")
    return runs


def available_pairs_message(metrics_dir, model, tags=None, limit=12):
    """Return a sentence listing the sample and topology pairs a model does hold runs for."""
    pairs = sorted({(run["sample"], run["topology"])
                    for path in glob.glob(os.path.join(metrics_dir, "*", "**", model, "*", "metrics.csv"),
                                          recursive=True)
                    for run in [run_of_path(path, metrics_dir)]
                    if matches(run["tag"], tags)})
    if not pairs:
        return f"No run of the model '{model}' was found at all."
    listed = ", ".join(f"{sample}/{topology}" for sample, topology in pairs[:limit])
    more = f" and {len(pairs) - limit} more" if len(pairs) > limit else ""
    return f"The sample/topology pairs available are {listed}{more}."


def hyperparameter_columns(runs, model, grid_file):
    """Return the names of the columns of the runs that hold a hyperparameter rather than a metric.

    The names are read from the grid file, which is where they are defined, and cross-checked against the
    metric naming of 'Compute_metrics.py'. The grid file is the one in use now and the runs may predate an
    edit to it, so a disagreement falls back on the naming rather than dropping a column silently."""
    columns = [column for column in runs.columns if column not in RUN_COLUMNS]
    inferred = [column for column in columns if not is_metric_name(column)]

    if grid_file is None or not os.path.exists(grid_file):
        return inferred

    with open(grid_file) as f:
        grids = json.load(f)
    if model not in grids:
        print(f"Warning: no hyperparameter grid given for the model '{model}' in {grid_file}, whose "
              f"grids are those of {list(grids)}. The hyperparameters are read from the column names.")
        return inferred

    from_grid = list(grid_of(grids[model])[0])
    missing = [name for name in from_grid if name not in columns]
    extra = [name for name in inferred if name not in from_grid]
    if missing or extra:
        print(f"Warning: the hyperparameters of {grid_file} ({from_grid}) do not match the columns of "
              f"the metrics files ({inferred}): the grid may have changed since these runs were "
              f"produced. The column names are used instead.")
        return inferred
    return from_grid


def tidy_hyperparameter(column):
    """Return a hyperparameter column with whole-valued numbers read back as integers.

    The hyperparameters travel through a csv file, so integers come back as floats and would be labelled
    'max_depth=5.0' on the plots. A column is only cast when every one of its values is whole, so that a
    hyperparameter scanned over both integer and fractional values (ex: gamma, over 0 and 0.5) keeps its
    fractional ones: casting it to the type of a single reference value, as 'Gather_metrics.py' does for
    the one row it keeps, would turn 0.5 into 0 here."""
    if not pd.api.types.is_float_dtype(column):
        return column
    if column.isna().any() or not np.isfinite(column).all():
        return column
    if (column == column.round()).all():
        return column.astype("int64")
    return column


def scanned_hyperparameters(runs, hyperparameters):
    """Return the hyperparameters taking more than one value among the selected runs.

    A hyperparameter left with a single value, whether because the grid only ever held one (ex: the XGB
    'subsample') or because a cut or a '--fix' narrowed the selection down to it, carries no information
    on a plot: it is never plotted against and never labels a line, and is reported once under the figure
    instead. The set is computed once on the whole selection rather than panel by panel, so that every
    panel of a figure shares the same lines and the same legend."""
    return [name for name in hyperparameters if runs[name].nunique(dropna=False) > 1]


def listed(values):
    """Return values as plain Python ones, so that they read as '0.5' and not 'np.float64(0.5)'."""
    return [value.item() if hasattr(value, "item") else value for value in sorted(values)]


def value_in_column(raw, column):
    """Return the value of a column matching the string a '--fix' was given, or None if there is none.

    The value is matched on its text first and on its number second, so that '--fix max_depth 5' matches
    the integer 5, '--fix gamma 0.5' the float 0.5, and '--fix gamma 0' the value 0 whichever of the two
    types it was read back as."""
    values = list(column.unique())
    for value in values:
        if str(value) == raw:
            return value
    try:
        wanted = float(raw)
    except (TypeError, ValueError):
        return None
    for value in values:
        try:
            if float(value) == wanted:
                return value
        except (TypeError, ValueError):
            continue
    return None


def apply_fixes(runs, fixes, hyperparameters, model):
    """Return the runs whose hyperparameters take the values given with '--fix', and those values."""
    fixed = {}
    for name, raw in fixes:
        if name in fixed:
            raise ValueError(f"The hyperparameter '{name}' is fixed more than once.")
        if name not in hyperparameters:
            close = difflib.get_close_matches(name, hyperparameters, n=3, cutoff=0.5)
            raise ValueError(f"'{name}' is not a hyperparameter of the model '{model}', whose "
                             f"hyperparameters are {hyperparameters}."
                             + (f" Did you mean {close}?" if close else ""))
        value = value_in_column(raw, runs[name])
        if value is None:
            raise ValueError(f"No run of the model '{model}' has {name}={raw}: the values '{name}' takes "
                             f"in the selected runs are {listed(runs[name].unique())}.")
        runs = runs[runs[name] == value]
        fixed[name] = value
    return runs, fixed


def panel_columns(runs):
    """Return the cut fields that still take several values, and so need a panel each.

    The fields pinned down to a single value are the same for every run and are written once under the
    figure rather than repeated in every panel title. Keeping the tag among them matters: two tags are
    two different scans, with their own baselines, and must never share a panel."""
    return [column for column in CUT_COLUMNS if runs[column].nunique() > 1]


def panels_of(runs, columns):
    """Return an ordered {panel label: runs} mapping, one entry per value of the panel columns."""
    if not columns:
        return {"": runs}
    return {" / ".join(str(value) for value in keys) if isinstance(keys, tuple) else str(keys): group
            for keys, group in runs.groupby(columns, sort=True)}


def baseline_of(runs, metric):
    """Return the mean, minimum and maximum of the un-reweighted baseline of a metric, or None.

    The baseline is the same quantity taken without any reweighting. It is computed once per sample and
    topology by 'Compute_baseline_metrics.py' and merged by 'Compute_metrics.py' into every run it
    belongs to, so it is identical across the runs of a sample and topology, and the minimum and maximum
    are returned only because a selection spanning several samples or topologies still holds more than
    one of them. Runs of older tags, written before the baseline was hoisted out of 'Compute_metrics.py'
    and its projection directions seeded, instead carry a sliced Wasserstein baseline re-estimated from
    fresh random projections in every run, varying by a few percent between them: the spread is drawn as
    a band, which for those tags is the noise floor the differences between hyperparameter sets should be
    judged against."""
    column = BASELINE_PREFIX + metric
    if metric.startswith(BASELINE_PREFIX) or column not in runs.columns:
        return None
    values = runs[column].dropna()
    if values.empty:
        return None
    return values.mean(), values.min(), values.max()


def panel_grid(n_panels, share_metric_scale, panel_size=(5.0, 4.0)):
    """Return the figure and the axes of a roughly square grid holding one panel per selection.

    The panels always share their x axis, since every one of them is scanned over the same hyperparameter
    values, and share the metric axis unless told otherwise: metrics such as the total chi2/dof span
    orders of magnitude from one topology to the next, which a shared scale would flatten."""
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey=share_metric_scale,
                             figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
                             layout="constrained")
    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)
    return fig, list(axes.ravel()[:n_panels]), n_cols


def label_outer_panels(axes, n_cols, xlabel, ylabel):
    """Label the outer panels of a grid only, the inner ones sharing their axes with them."""
    for index, ax in enumerate(axes):
        if index >= len(axes) - n_cols or index + n_cols >= len(axes):
            ax.set_xlabel(xlabel)
        if index % n_cols == 0:
            ax.set_ylabel(ylabel)


def line_colours(n_lines, n_nuisance):
    """Return the colours of the lines of a panel.

    A single hyperparameter left to label the lines gives them a sequential colour map, so that the order
    of the legend is the order of the colours and a monotonic trend can be read off at a glance. Several
    of them give a qualitative map instead, there being no single order to follow."""
    if n_nuisance <= 1:
        return [plt.get_cmap("viridis")(position) for position in np.linspace(0.0, 0.85, max(n_lines, 1))]
    return [plt.get_cmap("tab10")(index % 10) for index in range(n_lines)]


def draw_lines(ax, runs, hyperparameter, metric, nuisance, combinations, colours, reduce_op=None):
    """Draw a metric against one hyperparameter, one line per combination of the other hyperparameters."""
    if reduce_op is not None:
        reduced = runs.groupby(hyperparameter)[metric].agg(reduce_op).sort_index()
        ax.plot(reduced.index, reduced.to_numpy(), marker="o", ms=4, lw=1.4, color=colours[0],
                label=f"{reduce_op} over {', '.join(nuisance)}" if nuisance else metric)
        return

    for index, combination in enumerate(combinations):
        selected = runs
        for name, value in zip(nuisance, combination):
            selected = selected[selected[name] == value]
        selected = selected.sort_values(hyperparameter)
        if selected.empty:
            continue
        ax.plot(selected[hyperparameter], selected[metric], marker="o", ms=4, lw=1.4,
                color=colours[index], label=", ".join(str(value) for value in combination) or metric)


def draw_baseline(ax, baseline):
    """Draw the un-reweighted baseline of a metric as a dashed line, with its spread over the runs."""
    mean, minimum, maximum = baseline
    ax.axhline(mean, ls="--", lw=1.0, color="0.4",
               label="Original" if maximum <= minimum else "Original (spread over runs)")
    if maximum > minimum:
        ax.axhspan(minimum, maximum, color="0.7", alpha=0.3, lw=0)


def draw_heatmap(ax, runs, x_hyperparameter, y_hyperparameter, metric, x_values, y_values, cmap, norm,
                 number_format, reduce_op):
    """Draw a metric over two hyperparameters as a heatmap with the value printed in every cell.

    The cells sit at integer positions and are labelled with the values themselves, so that a grid scanned
    over irregular or logarithmically spaced values (ex: a learning rate over 0.01, 0.05, 0.1 and 0.2)
    gives cells of equal width. Grid points no run completed for stay empty."""
    grid = (runs.pivot_table(index=y_hyperparameter, columns=x_hyperparameter, values=metric,
                             aggfunc=reduce_op or "mean")
                .reindex(index=y_values, columns=x_values))
    values = np.ma.masked_invalid(grid.to_numpy(dtype=float))

    edges_x = np.arange(len(x_values) + 1) - 0.5
    edges_y = np.arange(len(y_values) + 1) - 0.5
    ax.set_facecolor("white")
    mesh = ax.pcolormesh(edges_x, edges_y, values, cmap=cmap, norm=norm, edgecolors="lightgrey",
                         linewidth=0.5)

    colour_map = plt.get_cmap(cmap)
    for row, column in np.ndindex(values.shape):
        if values.mask.ndim and values.mask[row, column]:
            continue
        value = values[row, column]
        red, green, blue, _ = colour_map(norm(value))
        ax.text(column, row, format(value, number_format), ha="center", va="center", fontsize=7,
                color="white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black")

    ax.set_xticks(range(len(x_values)), [str(value) for value in x_values])
    ax.set_yticks(range(len(y_values)), [str(value) for value in y_values])
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    if any(len(str(value)) > 6 for value in x_values):
        ax.tick_params(axis="x", labelrotation=45)
    return mesh


def metric_limits(panels, metric, log_metric):
    """Return the range a metric spans over every panel, to give them a shared scale."""
    values = pd.concat([panel[metric] for panel in panels.values()]).dropna()
    if log_metric:
        values = values[values > 0]
    if values.empty:
        return None, None
    return float(values.min()), float(values.max())


def print_available(runs, model, hyperparameters, scanned, metrics, metrics_dir):
    """Print the tags, samples, topologies, hyperparameters and metrics the selected runs hold."""
    print(f"Tags holding per-run metrics csv files: {', '.join(tags_with_metrics(metrics_dir))}")
    print(f"Runs of the model '{model}' matching the cuts: {len(runs)}")
    print()
    for column, plural in zip(CUT_COLUMNS, ("Tags", "Samples", "Topologies")):
        print(f"{plural:12s} {', '.join(sorted(runs[column].unique()))}")
    print()

    print("Hyperparameters (the values they take in the selected runs):")
    for name in hyperparameters:
        values = sorted(runs[name].unique())
        kind = "scanned " if name in scanned else "constant"
        print(f"  {kind}  {name}: {', '.join(str(value) for value in values)}")
    print()

    print("Metrics:")
    baselines = [metric for metric in metrics if metric.startswith(BASELINE_PREFIX)]
    for family in ("SWD", "p_value", "chi2_dof", "chi2_p_value",
                   "data_SWD", "data_p_value", "data_chi2_dof", "data_chi2_p_value"):
        listed = [metric for metric in metrics
                  if not metric.startswith(BASELINE_PREFIX) and metric.startswith(f"{family}_")]
        if listed:
            print(f"  {family + ':':20s}{', '.join(listed)}")
    training = [metric for metric in metrics if metric in TRAINING_METRICS]
    if training:
        print(f"  {'training:':16s}{', '.join(training)}")
    if baselines:
        print(f"  {'Original_*:':16s}the same {len(baselines)} metrics without any reweighting, "
              f"drawn as a reference line")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Plot a metric of the fine-tuning runs against one or two of the hyperparameters "
                    "they were run with.")

    argparser.add_argument("--model", type=str, required=True,
                           help="The reweighting model to plot the runs of. Exactly one model per "
                                "figure, the hyperparameters differing from one model to the next.")
    argparser.add_argument("--metric", type=str,
                           help="The metric to plot (ex: SWD_reweighting, chi2_dof_8D, val_auc). "
                                "Required unless '--list' is given, which prints every metric available.")
    argparser.add_argument("--hyperparameters", nargs="+", metavar="NAME",
                           help="The hyperparameter(s) to plot the metric against: one gives a line plot, "
                                "with one line per combination of the hyperparameters that are not "
                                "plotted against, and two give an annotated heatmap, the first on the x "
                                "axis and the second on the y axis. Required unless '--list' is given.")
    argparser.add_argument("--fix", action="append", nargs=2, default=[], metavar=("NAME", "VALUE"),
                           help="Keep only the runs whose hyperparameter NAME takes the value VALUE, to "
                                "cut down the number of lines or the runs a heatmap cell holds. Can be "
                                "given once per hyperparameter.")
    argparser.add_argument("--reduce", choices=("min", "max", "mean"),
                           help="Aggregate over the hyperparameters that are neither plotted against nor "
                                "fixed, instead of drawing one line per combination of them. Required in "
                                "heatmap mode when such hyperparameters are left, each cell otherwise "
                                "holding several runs.")

    argparser.add_argument("--metrics_dir", type=str, default="saved_metrics",
                           help="Directory holding the metrics of every run, as written by "
                                "'Compute_metrics.py'.")
    argparser.add_argument("--grid_file", type=str, default="hyperparameter_grids.json",
                           help="Json file holding the hyperparameter grid of every model, which the "
                                "names of the hyperparameter columns are read from. The column names are "
                                "used instead if it is missing or no longer matches the runs.")

    argparser.add_argument("--tag", nargs="+", metavar="PATTERN",
                           help="Output tag(s) to read the runs of, as shell-style patterns. Defaults to "
                                "every tag holding per-run metrics files.")
    argparser.add_argument("--sample", nargs="+", metavar="PATTERN",
                           help="Sample(s) to keep, as their path under the tag and as shell-style "
                                "patterns whose '*' spans '/' as well (ex: ND280/CH/numu, 'ND280/*'). "
                                "Defaults to every sample.")
    argparser.add_argument("--topology", nargs="+", metavar="PATTERN",
                           help="Topology(ies) to keep, as shell-style patterns (ex: CC0pi, 'CC*'). "
                                "Defaults to every topology.")

    argparser.add_argument("--output_file", type=str,
                           help="File to write the figure to, its extension giving the format. Required "
                                "unless '--list' is given.")
    argparser.add_argument("--list", action="store_true",
                           help="Print the tags, samples, topologies, hyperparameters and metrics "
                                "available for the given model and cuts, and exit without plotting.")
    argparser.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                           help="Whether to show the un-reweighted 'Original_' twin of the metric as a "
                                "reference, for the metrics that have one.")
    argparser.add_argument("--log_metric", action=argparse.BooleanOptionalAction, default=False,
                           help="Whether to put the metric on a logarithmic scale (the y axis of a line "
                                "plot, the colour scale of a heatmap).")
    argparser.add_argument("--log_x", action=argparse.BooleanOptionalAction, default=False,
                           help="Whether to put the scanned hyperparameter on a logarithmic x axis. Line "
                                "plots only: the axes of a heatmap are categorical.")
    argparser.add_argument("--share_scale", action=argparse.BooleanOptionalAction, default=False,
                           help="Whether the panels share the metric scale. Off by default: the metrics "
                                "of the different topologies routinely differ by an order of magnitude, "
                                "and a single large panel then flattens every other one to a line. Turn "
                                "it on to read the panels against each other.")
    argparser.add_argument("--cmap", type=str, default="viridis", help="Colour map of the heatmap.")
    argparser.add_argument("--number_format", type=str, default=".3g",
                           help="Format of the numbers printed in the heatmap cells.")
    argparser.add_argument("--title", type=str,
                           help="Title of the figure. Defaults to one built from the model, the metric "
                                "and the hyperparameters.")
    args = argparser.parse_args()

    if not args.list:
        missing = [name for name, value in (("--metric", args.metric),
                                            ("--hyperparameters", args.hyperparameters),
                                            ("--output_file", args.output_file)) if not value]
        if missing:
            argparser.error(f"the following arguments are required: {', '.join(missing)} "
                            "(or '--list', to print what is available)")
        if len(args.hyperparameters) > 2:
            argparser.error(f"'--hyperparameters' takes one hyperparameter (line plot) or two (heatmap), "
                            f"not {len(args.hyperparameters)}: {args.hyperparameters}")

    runs = load_runs(args.metrics_dir, args.model, tags=args.tag, samples=args.sample,
                     topologies=args.topology)

    hyperparameters = hyperparameter_columns(runs, args.model, args.grid_file)
    metrics = [column for column in runs.columns
               if column not in RUN_COLUMNS and column not in hyperparameters]
    for name in hyperparameters:
        runs[name] = tidy_hyperparameter(runs[name])

    runs, fixed = apply_fixes(runs, args.fix, hyperparameters, args.model)
    if runs.empty:
        raise ValueError(f"No run of the model '{args.model}' is left by the cuts and the fixed "
                         f"hyperparameters {fixed}.")
    scanned = scanned_hyperparameters(runs, hyperparameters)

    if args.list:
        print_available(runs, args.model, hyperparameters, scanned, metrics, args.metrics_dir)
        raise SystemExit(0)

    if args.metric not in metrics:
        close = difflib.get_close_matches(args.metric, metrics, n=3, cutoff=0.5)
        raise ValueError(f"The metric '{args.metric}' is not held by the runs of the model "
                         f"'{args.model}'." + (f" Did you mean {close}?" if close else "")
                         + " Run with '--list' to print every metric available.")

    for name in args.hyperparameters:
        if name not in hyperparameters:
            close = difflib.get_close_matches(name, hyperparameters, n=3, cutoff=0.5)
            raise ValueError(f"'{name}' is not a hyperparameter of the model '{args.model}', whose "
                             f"hyperparameters are {hyperparameters}."
                             + (f" Did you mean {close}?" if close else "")
                             + " Metrics are given with '--metric'.")
        if name in fixed:
            raise ValueError(f"The hyperparameter '{name}' cannot be both plotted against "
                             f"('--hyperparameters') and fixed ('--fix').")
        if name not in scanned:
            raise ValueError(f"The hyperparameter '{name}' takes the single value "
                             f"{runs[name].iloc[0]} in the selected runs, so there is nothing to plot "
                             f"against. The hyperparameters that vary are {scanned}.")

    nuisance = [name for name in scanned if name not in args.hyperparameters]
    if len(args.hyperparameters) == 2 and nuisance and args.reduce is None:
        values = ", ".join(f"{name}: {listed(runs[name].unique())}" for name in nuisance)
        raise ValueError(f"The hyperparameter(s) {nuisance} still vary, so every cell of the heatmap "
                         f"would hold several runs. Pin them with '--fix NAME VALUE' ({values}) or "
                         f"aggregate over them with '--reduce min|max|mean'.")

    dropped = runs[args.metric].isna()
    if dropped.any():
        print(f"Warning: {int(dropped.sum())} run(s) hold no value for '{args.metric}' and are left out.")
        runs = runs[~dropped]
    if args.log_metric:
        non_positive = runs[args.metric] <= 0
        if non_positive.any():
            print(f"Warning: {int(non_positive.sum())} run(s) have {args.metric} <= 0 and cannot be "
                  f"shown on a logarithmic scale.")
    if args.log_x and len(args.hyperparameters) == 2:
        print("Warning: '--log_x' is ignored in heatmap mode, whose axes are categorical.")

    panel_cols = panel_columns(runs)
    panels = panels_of(runs, panel_cols)
    if len(panels) > 30:
        print(f"Warning: {len(panels)} panels; consider narrowing '--sample' or '--topology'.")

    duplicated = runs.duplicated(subset=panel_cols + scanned, keep=False)
    if duplicated.any() and args.reduce is None:
        print(f"Warning: {int(duplicated.sum())} runs share a hyperparameter set within a panel "
              f"(hp_ids {sorted(runs.loc[duplicated, 'hp_id'].unique())}); their metric is averaged.")

    mh.style.use("DUNE")
    use_latex_default_font()
    plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 9,
                         "figure.titlesize": 13})

    fig, axes, n_cols = panel_grid(len(panels), args.share_scale)

    pinned = [f"{column}={runs[column].iloc[0]}" for column in CUT_COLUMNS if column not in panel_cols]
    constant = [f"{name}={runs[name].iloc[0]}" for name in hyperparameters
                if name not in scanned and name not in fixed]
    subtitle = "\n".join(textwrap.wrap(" | ".join(filter(None, [
        ", ".join(pinned),
        f"{len(runs)} runs",
        f"fixed: {', '.join(f'{name}={value}' for name, value in fixed.items())}" if fixed else "",
        f"constant: {', '.join(constant)}" if constant else "",
    ])), width=110))

    if len(args.hyperparameters) == 1:
        hyperparameter = args.hyperparameters[0]
        combinations = sorted({tuple(row) for row in runs[nuisance].to_numpy()}) if nuisance else [()]
        colours = line_colours(len(combinations), len(nuisance))
        x_values = sorted(runs[hyperparameter].unique())

        for ax, (label, panel) in zip(axes, panels.items()):
            draw_lines(ax, panel, hyperparameter, args.metric, nuisance, combinations, colours,
                       reduce_op=args.reduce)
            baseline = baseline_of(panel, args.metric) if args.baseline else None
            if baseline is not None:
                draw_baseline(ax, baseline)
            ax.set_title(label)
            if args.log_x:
                ax.set_xscale("log")
            elif len(x_values) <= 12:
                ax.set_xticks(x_values, [str(value) for value in x_values])
            if args.log_metric:
                ax.set_yscale("log")
            ax.grid(alpha=0.25, lw=0.5)

        label_outer_panels(axes, n_cols, hyperparameter, metric_label(args.metric))
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title=", ".join(nuisance) or None, loc="outside right upper",
                       frameon=False, ncol=1 + len(labels) // 14)
    else:
        x_hyperparameter, y_hyperparameter = args.hyperparameters
        x_values = sorted(runs[x_hyperparameter].unique())
        y_values = sorted(runs[y_hyperparameter].unique())
        minimum, maximum = metric_limits(panels, args.metric, args.log_metric) if args.share_scale \
            else (None, None)

        for ax, (label, panel) in zip(axes, panels.items()):
            if args.share_scale:
                norm = LogNorm(minimum, maximum) if args.log_metric else Normalize(minimum, maximum)
            else:
                low, high = metric_limits({label: panel}, args.metric, args.log_metric)
                norm = LogNorm(low, high) if args.log_metric else Normalize(low, high)
            mesh = draw_heatmap(ax, panel, x_hyperparameter, y_hyperparameter, args.metric, x_values,
                                y_values, args.cmap, norm, args.number_format, args.reduce)
            baseline = baseline_of(panel, args.metric) if args.baseline else None
            ax.set_title(label if baseline is None
                         else f"{label}\nOriginal = {format(baseline[0], args.number_format)}")
            if not args.share_scale:
                fig.colorbar(mesh, ax=ax, label=metric_label(args.metric))
        label_outer_panels(axes, n_cols, x_hyperparameter, y_hyperparameter)
        if args.share_scale:
            fig.colorbar(mesh, ax=axes, label=metric_label(args.metric), shrink=0.8)

    title = args.title or (f"{args.model}: {metric_label(args.metric)} against "
                           f"{' and '.join(args.hyperparameters)}")
    fig.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=11)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote the {len(panels)}-panel figure of {len(runs)} runs to {args.output_file}")
