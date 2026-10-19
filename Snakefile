configfile: "config.yaml"

from itertools import chain
import glob
import hashlib
import json
import math
import os
import re

import pyarrow.parquet as pq

from Constants import DATA_N_EVENTS, DEFAULT_DATA_N_EVENTS, TOPOLOGY_CODES, TOPOLOGY_COLUMN, data_n_events_of
from List_hyperparameters import grids_of_file, labelled_grid_of, labels_of
from Param_sets import ALL_PARAMS_SET, REWEIGHTING_SET
from Train_predict import model_extension

ENV = "environment.yaml"
TAG = config["output"]["tag"]
MODELS = config["models"]["model_list"]
GRID_FILE = config["models"]["grid_file"]
HP_LABELS = {model: labels_of(GRID_FILE, model) for model in MODELS}
# The {label: hyperparameters} dictionary of the grid of every model.
HP_GRIDS = {model: labelled_grid_of(grids_of_file(GRID_FILE, model)) for model in MODELS}
EXCLUDED_HPS_FILE = config["models"].get("excluded_hps_file")
EXCLUDED_HPS = {}
if EXCLUDED_HPS_FILE:
    with open(EXCLUDED_HPS_FILE) as f:
        EXCLUDED_HPS = json.load(f)

def values_match(value, wanted):
    """Whether a hyperparameter value is the one an exclusion asks for, or one of a list of them.
    Numbers are compared as numbers, so that 0.1 and 1e-1 are the same value."""
    wanted = wanted if isinstance(wanted, list) else [wanted]
    def same(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=0)
        return a == b
    return any(same(value, w) for w in wanted)

def excluded_labels(model, entry):
    """Return the labels of the grid points of a model an exclusion entry leaves out.

    An entry is either the label of a single grid point, or a {hyperparameter: value or [values]}
    dictionary leaving out every grid point whose hyperparameters all match it, ex
    {"learning_rate": 0.01, "max_depth": 9, "lambda_frac": [0.1, 0.3]}. A dictionary naming a
    hyperparameter the grid of the model does not scan is raised rather than left to match
    nothing, as it is most likely a typo."""
    grid = HP_GRIDS[model]
    if isinstance(entry, str):
        return {entry}
    scanned = set().union(*(hyperparameters.keys() for hyperparameters in grid.values()))
    unknown = set(entry) - scanned
    if unknown:
        raise ValueError(f"The exclusion {entry} of the model '{model}' in {EXCLUDED_HPS_FILE} names "
                         f"the hyperparameter(s) {sorted(unknown)}, which its grid does not have. "
                         f"The grid has {sorted(scanned)}.")
    return {label for label, hyperparameters in grid.items()
            if all(values_match(hyperparameters[name], wanted) for name, wanted in entry.items())}

def hp_labels_of(sample, topology, model):
    """The grid points of a model that are run for a given sample/topology: the whole grid but
    those listed in the excluded_hps_file. A '*' sample or topology key applies its exclusions to
    every sample or topology, on top of those listed under their own names."""
    excluded = set()
    for sample_key in (sample, "*"):
        for topology_key in (topology, "*"):
            for entry in EXCLUDED_HPS.get(sample_key, {}).get(topology_key, {}).get(model, []):
                excluded |= excluded_labels(model, entry)
    return [hp_id for hp_id in HP_LABELS[model] if hp_id not in excluded]
SELECTION_METRIC = config["models"]["selection_metric"]
MODES = config["analysis"]["modes"]
TOPOLOGIES = config["analysis"]["topologies"]
# Optional fraction of each input file's events to read in; unset (None) reads every event.
DOWNSAMPLING = config["analysis"].get("downsampling")
# Optional cap on the number of events a model is trained on, as a multiple of the data size of its
# sample and topology (see 'max_training_events'); unset (None) trains on every event.
MAX_TRAINING_EVENTS_FACTOR = config["analysis"].get("max_training_events_factor")

RUNS = range(config["swd_bootstrapping"]["runs"])
# Number of data-sized subsamples the data-scale SWD and p-value of every run are averaged over.
N_SUBSAMPLES = config["data_scale"]["n_subsamples"]

def data_n_events_for(wildcards):
    """Return the total number of raw events the real downstream analysis will use for a sample."""
    return DATA_N_EVENTS.get(wildcards.sample, DEFAULT_DATA_N_EVENTS)

def topology_data_n_events(sample, topology):
    """Return the number of events the real downstream analysis will use for one topology of a
    sample: the sample's data size split across its topologies by their target rates."""
    return data_n_events_of(DATA_N_EVENTS.get(sample, DEFAULT_DATA_N_EVENTS),
                            TARGET_TOPOLOGY_RATES_FILE.format(sample=sample), topology)

def max_training_events(sample, topology):
    """Return the number of events the training (and validation) samples of a sample and topology
    are capped at: MAX_TRAINING_EVENTS_FACTOR times the topology's data size (see
    'topology_data_n_events'). None when no factor is given, which trains on every event."""
    if MAX_TRAINING_EVENTS_FACTOR is None:
        return None
    return int(round(MAX_TRAINING_EVENTS_FACTOR * topology_data_n_events(sample, topology)))

def training_seed(sample, topology):
    """Return the seed the training sample of a sample and topology is capped with."""

    return int(hashlib.sha256(f"{sample}/{topology}".encode()).hexdigest()[:8], 16)

def swd_direction_seed(sample, topology):
    """Return the seed the SWD projection directions of a sample and topology are drawn from.

    The baseline job and every 'compute_metrics' job of a sample and topology are given this one
    seed, so that they measure their distances along the same directions and the hyperparameter
    sets can be compared without the noise of the estimator itself. Its own prefix keeps it from
    coinciding with 'training_seed'."""

    return int(hashlib.sha256(f"swd_directions/{sample}/{topology}".encode()).hexdigest()[:8], 16)

def data_subsample_seed(sample, topology):
    """Return the seed the data-sized subsamples of a sample and topology are drawn from.

    As with 'swd_direction_seed', the baseline job and every 'compute_metrics' job of a sample and
    topology are given this one seed, so that they are all evaluated on the same events."""

    return int(hashlib.sha256(f"data_subsamples/{sample}/{topology}".encode()).hexdigest()[:8], 16)

# ---------------------------------------------------------------------------
# Resource requests
# The jobs downstream of 'initialize_analysis' are sized from the number of events of their sample
# and topology, rather than from per-topology tables, so that the requests carry over to samples
# whose topologies are of different sizes. The constants are calibrated from the benchmarks of the
# scarf0 run (ND280/CH/numu, 20M events per generator); memory is in MB and runtimes in minutes.
# Memory is not grown by the retry attempt, since retries are for timeouts, which more memory does
# not help.
# ---------------------------------------------------------------------------

def events_per_generator(sample, topology, split):
    """Return the {'original': n, 'target': n} numbers of events of a split ('train', 'val' or
    'test') of a sample and topology, read from the parquet metadata."""
    code = TOPOLOGY_CODES[topology]
    counts = {}
    for generator in ("original", "target"):
        partition = os.path.join(BASE_SAMPLES_DIR.format(sample=sample),
                                 f"{generator}_{split}.parquet", f"{TOPOLOGY_COLUMN}={code}")
        counts[generator] = sum(pq.read_metadata(f).num_rows
                                for f in glob.glob(os.path.join(partition, "*.parquet")))
    return counts

def events_of(sample, topology, split):
    """Return the number of events of the larger of the original and target samples of a split."""
    return max(events_per_generator(sample, topology, split).values())

def capped_training_events(sample, topology):
    """Return the number of events a model is trained on, after the 'max_training_events' cap."""
    n = events_of(sample, topology, "train")
    cap = max_training_events(sample, topology)
    return n if cap is None else min(n, cap)

def round_up(value, step):
    return int(-(-value // step) * step)

# XGB training time is the number of trees times a cost per tree proportional to the number of
# training events: 1.05-1.22 us per event per tree (median, depth 3-9), 1.47-1.61 us at the 99th
# percentile and 1.89 us at most, so 1.8 us is requested. Early stopping rarely helps: over half
# the runs went to the full n_estimators, so every run is given the time for all of them. The overhead covers
# loading the samples and saving the model (5-16 minutes).
XGB_US_PER_EVENT_PER_TREE = 1.8
XGB_OVERHEAD_MIN = 20
MIN_TRAIN_RUNTIME = 60

def train_runtime(wildcards, attempt):
    if wildcards.model == "binning":  # under a minute in every benchmark
        return MIN_TRAIN_RUNTIME * attempt
    n_trees = HP_GRIDS[wildcards.model][wildcards.hp_id]["n_estimators"]
    n_events = capped_training_events(wildcards.sample, wildcards.topology)
    minutes = XGB_OVERHEAD_MIN + n_trees * n_events * XGB_US_PER_EVENT_PER_TREE * 1e-6 / 60
    return round_up(max(minutes, MIN_TRAIN_RUNTIME), 30) * attempt

# The whole training and validation samples are read before being capped, so memory follows the
# uncapped number of events: the peak was ~0.6 KB per event (3.7-4.1 GB for CC0pi's 6.5M), with
# depth-9 models adding up to ~1.3 GB on top for the smallest topologies. That was with the 5
# reweighting parameters of scarf0 and complete_0, of which the samples read are made: the
# per-event memory is scaled by the number of reweighting parameters over those 5.
TRAIN_MEM_MB_PER_EVENT = 0.5e-3
TRAIN_MEM_CALIBRATION_PARAMS = 5

def train_mem_mb(wildcards):
    n_events = events_of(wildcards.sample, wildcards.topology, "train")
    per_event = TRAIN_MEM_MB_PER_EVENT * len(PARAM_SET_DICT[REWEIGHTING_SET]) / TRAIN_MEM_CALIBRATION_PARAMS
    return round_up(max(4000, 2000 + per_event * n_events), 500)

# Metric jobs compute their metrics on the whole test sample, which the training cap does not touch,
# and again on each of the N_SUBSAMPLES data-sized subsamples. A subsample is the topology's data
# size, or the whole test sample when that is smaller (as for the CH2_H samples), so the subsamples
# can cost many times the full-sample metrics. Over the 2088 metric and baseline jobs of complete_0
# and the 3100 of scarf0, a job took 0.95 ms per test event (original + target), plus 0.40 ms per
# event of every subsample (original + target), plus 5.4 ns per tree per original test event for
# the predictions of an XGB model, at most 1.4-1.7x that for 99% of the jobs. The tree count is the
# grid's n_estimators, an upper bound: early stopping can only leave fewer. Requesting 10 minutes
# + 1.7x the fit, at least an hour, put every complete_0 job within its first attempt.
# Loading a depth-9 model took 'compute_metrics' up to 5.9 GB whatever the topology, against
# ~1 GB plus ~0.7 KB per test event for 'compute_baseline_metrics', which loads none.
def metrics_runtime(wildcards, attempt):
    n = events_per_generator(wildcards.sample, wildcards.topology, "test")
    n_subsample = min(int(round(topology_data_n_events(wildcards.sample, wildcards.topology))),
                      min(n.values()))
    # 'compute_baseline_metrics' has no model, and a binning model no trees.
    model = wildcards.get("model")
    n_trees = HP_GRIDS[model][wildcards.hp_id].get("n_estimators", 0) if model else 0
    seconds = (950e-6 * (n["original"] + n["target"])
               + 405e-6 * N_SUBSAMPLES * 2 * n_subsample
               + 5.4e-9 * n_trees * n["original"])
    return round_up(max(60, 10 + 1.7 * seconds / 60), 30) * attempt

def metrics_mem_mb(wildcards):
    n_events = events_of(wildcards.sample, wildcards.topology, "test")
    return round_up(7000 + 0.5e-3 * n_events, 500)

def baseline_metrics_mem_mb(wildcards):
    n_events = events_of(wildcards.sample, wildcards.topology, "test")
    return round_up(max(4000, 1500 + 1e-3 * n_events), 500)

# 'initialize_analysis' is sized from its input ROOT files (both generators). Over the 24 samples
# of the nominal_multipi run (1-18 GB of input), its peak memory was 0.56 GB + 0.585 GB per GB of
# input, at most 0.70 GB per GB for the large samples and 1.06 GB per GB for the 2 GB ones, and it
# took 5 minutes + 4.5 minutes per GB, at most 85 minutes (18 GB) and 9 minutes per GB for the
# small samples. Both requests leave ~30% or more over the worst sample of its size, and both grow
# with the retry attempt: there are few of these jobs, and every other job waits on them.
def init_input_gb(input):
    return sum(os.path.getsize(f) for f in input) / 1e9

def init_mem_mb(wildcards, input, attempt):
    return round_up(max(2000, 1000 + 800 * init_input_gb(input)), 500) * attempt

def init_runtime(wildcards, input, attempt):
    return round_up(max(60, 20 + 6 * init_input_gb(input)), 30) * attempt


# ---------------------------------------------------------------------------
# Sets of analysis parameters
# The metrics are computed for every set listed in the config file, plus the set of parameters 
# the models are reweighted on.
# ---------------------------------------------------------------------------

METRIC_SETS = dict(config["parameters"].get("metric_sets") or {})
# Sets written by the initialisation, i.e. every set but the reweighting one, whose samples are
# obtained by keeping the reweighting parameters of the 'all' samples.
PARAM_SETS = list(METRIC_SETS) + [REWEIGHTING_SET]

ORIG_DIR = config["inputs"]["original_dir"]
TARGET_DIR = config["inputs"]["target_dir"]

# ---------------------------------------------------------------------------
# Sample discovery
# Scan ORIG_DIR recursively for .root files; keep any whose leaf directory
# also contains a .root file under the same relative path in TARGET_DIR.
# SAMPLES is a list of relative directory paths, e.g. ["FHC/numu/H2O"].
# ---------------------------------------------------------------------------

def get_samples():
    original_files = glob.glob(os.path.join(ORIG_DIR, "**/*.root"), recursive=True)
    seen = set()
    samples = []
    rejected = []
    for of in original_files:
        rel = os.path.relpath(os.path.dirname(of), ORIG_DIR)
        if rel in seen:
            continue
        seen.add(rel)

        orig_matches = glob.glob(os.path.join(ORIG_DIR, rel, "*.root"))
        target_matches = glob.glob(os.path.join(TARGET_DIR, rel, "*.root"))

        if len(orig_matches) == 1 and len(target_matches) == 1:
            samples.append(rel)
        else:
            rejected.append((rel, len(orig_matches), len(target_matches)))

    if rejected:
        print(f"Warning: rejected {len(rejected)} leaf sample(s) with != 1 ROOT file:")
        for rel, n_orig, n_target in rejected:
            print(f"  {rel}: {n_orig} original file(s), {n_target} target file(s)")

    return samples

SAMPLES = get_samples()

# Get union of all parameters
def get_all_params():
    all_sets = chain(
        config["parameters"]["reweighting"],
        config["parameters"]["extra"],
        *METRIC_SETS.values(),
    )
    return list(dict.fromkeys(all_sets))

def get_param_set_dict():
    dict = METRIC_SETS.copy()
    dict[ALL_PARAMS_SET] = get_all_params()
    dict[REWEIGHTING_SET] = config["parameters"]["reweighting"]
    return dict

PARAM_SET_DICT = get_param_set_dict()

# ---------------------------------------------------------------------------
# Input functions – resolve the single .root file inside a leaf directory.
# ---------------------------------------------------------------------------

def original_file_for(wildcards):
    files = glob.glob(os.path.join(ORIG_DIR, wildcards.sample, "*.root"))
    return files[0]

def target_file_for(wildcards):
    files = glob.glob(os.path.join(TARGET_DIR, wildcards.sample, "*.root"))
    return files[0]

# ---------------------------------------------------------------------------
# Per-sample path helpers
# ---------------------------------------------------------------------------

BASE_SAMPLES_DIR = f"saved_samples/{TAG}/{{sample}}/"
INIT_SWD_DIR     = f"saved_swd_distribution/{TAG}/{{sample}}/{{topology}}/"
ALL_MODELS_DIR   = f"saved_models/{TAG}/{{sample}}/{{topology}}/"
BASE_MODEL_DIR   = ALL_MODELS_DIR + f"{{model}}/"
MODEL_DIR        = BASE_MODEL_DIR + f"{{hp_id}}/"
FINAL_MODEL_DIR  = f"saved_final_models/{TAG}/{{sample}}/{{topology}}/"
ALL_METRICS_DIR  = f"saved_metrics/{TAG}/{{sample}}/{{topology}}/"
BASE_METRICS_DIR = f"saved_metrics/{TAG}/{{sample}}/{{topology}}/{{model}}/"
METRICS_DIR      = BASE_METRICS_DIR + f"{{hp_id}}/"
FIG_DIR          = f"saved_figures/{TAG}/{{sample}}/{{topology}}/"
HPS_DIR          = f"hps/{TAG}/"
CONFIGS_DIR      = f"configs/{TAG}/"
TENSORBOARD_DIR  = f"TensorBoard/{TAG}/{{sample}}/{{topology}}/"
TENSORBOARD_RUN_DIR = TENSORBOARD_DIR + "{model}/{hp_id}/"

# Benchmarks hold the peak memory and wall time of every submitted job, and are what the
# resource requests of the rules are calibrated from. They mirror the wildcards of the rule
# that wrote them.
BASE_BENCH_DIR   = f"benchmarks/{TAG}/{{sample}}/"
BENCH_DIR   = BASE_BENCH_DIR + "{topology}/"

# The un-reweighted baseline every run of a sample and topology is read against. It holds no
# {model}/{hp_id}, so it is not reached by the per-run metrics 'Gather_metrics.py' and
# 'Plot_hyperparameters.py' collect.
BASELINE_METRICS_FILE = ALL_METRICS_DIR + "baseline_metrics.csv"

# The bootstrapped SWD distribution of a set of parameters, whichever rule wrote its samples.
def swd_distribution_file(param_set):
    return INIT_SWD_DIR + f"{param_set}/swd_distribution_{param_set}.npy"

def model_file(model_name):
    return FINAL_MODEL_DIR.format(sample="{sample}", topology="{topology}") \
        + f"{model_name}/" \
        + f"{model_name}{model_extension(model_name)}"

# Command line arguments naming a set of parameters and the path it is given as, as the scripts
# computing the metrics take their bootstrapped SWD distributions ('--swd_distribution 3D path/to/3D.npy
# --swd_distribution 8D path/to/8D.npy ...').
def named_paths_flags(flag, paths):
    return " ".join(f"{flag} {param_set} {path}" for param_set, path in paths.items())

def swd_distributions_of(param_sets):
    return {param_set: swd_distribution_file(param_set) for param_set in param_sets}

# Hyperparameter grid files are sample-independent (shared search space).
HPS_FILES = []
for model in MODELS:
    for hp_id in HP_LABELS[model]:
        HPS_FILES.append(os.path.join(HPS_DIR, f"{model}/{model}_hp_{hp_id}.json"))

# Wildcard constraint: sample paths contain only word characters and slashes.
# The topology wildcard is restricted to the topology names listed in the config file,
# which also removes the ambiguity with the (slash-containing) sample wildcard.
# The parameter set and model wildcards are likewise restricted to the configured ones.
# 'hp_id' labels a point of a hyperparameter grid and 'run_id' numbers a bootstrap run: they
# name a single directory each, and constraining them keeps them from swallowing a path
# separator and so from reaching into the (slash-containing) sample wildcard.
wildcard_constraints:
    sample="[^.]+",
    topology="|".join(re.escape(str(t)) for t in TOPOLOGIES),
    param_set="|".join(re.escape(str(s)) for s in PARAM_SETS),
    model="|".join(re.escape(str(m)) for m in MODELS),
    hp_id=r"[^/]+",
    run_id=r"\d+"

INCLUDED_TOPOLOGIES_FILE = f"saved_samples/{TAG}/{{sample}}/included_topologies.json"
TARGET_TOPOLOGY_RATES_FILE = f"saved_samples/{TAG}/{{sample}}/target_topology_rates.json"

def topologies_for_sample(sample):
    """Return the topologies of the config file that hold enough events in a given sample."""
    with open(checkpoints.initialize_analysis.get(sample=sample).output.included_topologies) as f:
        included_topologies = json.load(f)
    
    return included_topologies

def all_metrics_files(wildcards):
    return [
        f"saved_metrics/{TAG}/{sample}/{topology}/metrics.json"
        for sample in SAMPLES
        for topology in topologies_for_sample(sample)
    ]

# The triangle plots 'Make_plots.py' writes, as 'triangle_{name}.pdf'.
TRIANGLE_PLOTS = ["original", "target", "ratio", "pull"]

# The p-value heatmaps of the whole tag: one per model (and the original sample, before
# reweighting) and parameter set.
HEATMAP_DIR = f"saved_figures/{TAG}/pvalue_heatmaps/"
HEATMAP_MODELS = MODELS + ["Original"]
HEATMAP_FILES = [HEATMAP_DIR + f"pvalue_heatmap_{model}_{param_set}.pdf"
                 for model in HEATMAP_MODELS for param_set in PARAM_SETS]

def all_figure_dirs(wildcards):
    return [
        f"saved_figures/{TAG}/{sample}/{topology}/{name}"
        for sample in SAMPLES
        for topology in topologies_for_sample(sample)
        for name in ["1Dhist.pdf", "1Dhist_data.pdf"] + [f"triangle_{plot}.pdf" for plot in TRIANGLE_PLOTS]
    ]

localrules: prepare_hps, aggregate_bootstrap, choose_models, save_configs, plot_pvalue_heatmaps

def scaled(base):
    """A resource that grows with the retry attempt: base, then 2*base, then 3*base."""
    return lambda wildcards, attempt: base * attempt

# Reads input files and identifies topologies with enough events to be split into training,
# validation, and test samples. The included topology names are written to a JSON file.
# Events in those topologies are split accordingly, stripped to the parameters listed in the
# config, and saved as Parquet files partitioned by topology. This is a checkpoint because the
# DAG depends on the included topologies, which are only known after it runs. This rule is run
# once for each sample.
checkpoint initialize_analysis:
    input:
        original_file=original_file_for,
        target_file=target_file_for

    params:
        modes=MODES,
        original_tree=config["inputs"]["original_tree"],
        target_tree=config["inputs"]["target_tree"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"],
        analysis_params=PARAM_SET_DICT[ALL_PARAMS_SET],
        topologies=TOPOLOGIES,
        downsampling=DOWNSAMPLING,
        samples_dir=BASE_SAMPLES_DIR

    output:
        last_sampled_files=directory(BASE_SAMPLES_DIR + "target_test.parquet"),
        included_topologies=INCLUDED_TOPOLOGIES_FILE,
        target_topology_rates=TARGET_TOPOLOGY_RATES_FILE

    threads: 1
    # See 'init_mem_mb' and 'init_runtime'.
    resources:
        mem_mb=init_mem_mb,
        runtime=init_runtime,
        # disk_mb=10000

    benchmark:
        BASE_BENCH_DIR + "initialize_analysis.tsv"

    conda:
        ENV

    shell:
        """
        python Init.py \
            --input_file_original {input.original_file} \
            --input_file_target {input.target_file} \
            --input_tree_original {params.original_tree} \
            --input_tree_target {params.target_tree} \
            --analysis_params {params.analysis_params} \
            --modes {params.modes} \
            --topologies {params.topologies} \
            --train_percentage {params.train_percentage} \
            --val_percentage {params.val_percentage} \
            --downsampling {params.downsampling} \
            --output_dir {params.samples_dir} \
            --topologies_file {output.included_topologies} \
            --target_topology_rates_file {output.target_topology_rates} \
        """

# Bootstraps swd_bootstrapping/n_samples pairs of data-sized samples from the target test sample,
# calculates the SWD between the two samples of each pair, and saves the SWD distribution
# as a .npy file: the null distribution the data-scale p-values of the metric rules are read
# against. This rule is run swd_bootstrapping/runs times for each sample/topology/
# parameter set combination. These are marked as temp, since they are later aggregated into 
# a single file.
rule run_bootstrap:
    input:
        target_test=BASE_SAMPLES_DIR + "target_test.parquet",
        target_topology_rates_file=TARGET_TOPOLOGY_RATES_FILE
    output:
        output_file=temp(INIT_SWD_DIR + "{param_set}/indiv_bootstrap/run_{run_id}.npy")
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"],
        param_set=lambda wc: PARAM_SET_DICT[wc.param_set],
        n_bootstrap=config["swd_bootstrapping"]["n_samples"],
        data_n_events=data_n_events_for
    # group: "swd"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=scaled(600),
        # disk_mb=10000
    benchmark:
        BENCH_DIR + "bootstrap_swd_{param_set}/run_bootstrap_{run_id}.tsv"
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.target_test} \
            --topology {wildcards.topology} \
            --param_set {params.param_set} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_bootstrap {params.n_bootstrap} \
            --n_directions {params.n_directions} \
            --data_n_events {params.data_n_events} \
            --target_topology_rates_file {input.target_topology_rates_file} \
            --random_seed {wildcards.run_id}
        """ 

# Local rule that aggregates the SWD distributions from run_bootstrap into a single .npy file.
# This rule is run once per sample/topology/parameter set combination
rule aggregate_bootstrap:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/{wc.topology}/{wc.param_set}/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=swd_distribution_file("{param_set}")
    conda:
        ENV
    shell:
        """
        python Aggregate.py \
            --input {input} \
            --output {output.final_file}
        """

# Local rule that writes a .json file for each hyperparameter set listed in the grid file. 
# These are marked as temp, since they contain no information not contained in the grid file.
rule prepare_hps:
    input:
        grid_file=GRID_FILE
    params:
        models=MODELS
    output:
        hps_files=HPS_FILES
    conda:
        ENV
    shell:
        """
        python List_hyperparameters.py \
            --grid_file {input.grid_file} \
            --models {params.models} \
            --output_dir {HPS_DIR}
        """

# Trains and saves a single model with a single hyperparameter set. This rule is run once 
# for each hyperparameter set for each model, for each sample/topology combination. The 
# trained model is saved as a .json file for XGB and unnormXGB, and as a .pkl file for binned 
# reweighting.
rule train_model:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        hparam_file=HPS_DIR + "{model}/{model}_hp_{hp_id}.json",
        # Read by 'max_training_events' for the cap, rather than by the script.
        target_topology_rates_file=TARGET_TOPOLOGY_RATES_FILE
    params:
        sample_dir=BASE_SAMPLES_DIR,
        model=lambda wc: wc.model,
        reweight_params=PARAM_SET_DICT[REWEIGHTING_SET],
        max_events=lambda wc: max_training_events(wc.sample, wc.topology),
        seed=lambda wc: training_seed(wc.sample, wc.topology)
    output:
        model_dir=directory(MODEL_DIR)
    threads: 2
    resources:
        mem_mb=train_mem_mb,
        runtime=train_runtime,
        # disk_mb=10000
    benchmark:
        BENCH_DIR + f"train_model_{{model}}/train_model_{{model}}_hp_{{hp_id}}.tsv"
    conda:
        ENV
    shell:
        """
        python Train_model.py \
            --sample_dir {params.sample_dir} \
            --topology {wildcards.topology} \
            --model {params.model} \
            --hyperparameters {input.hparam_file} \
            --reweighting_params {params.reweight_params} \
            --output_dir {output.model_dir} \
            --max_events {params.max_events} \
            --seed {params.seed}
        """

# Computes the metrics of the un-reweighted ('Original') baseline, which every trained model of a
# sample and topology is read against, and saves them to a .csv file. They depend on neither the
# model nor its hyperparameters, so this runs once per sample/topology combination instead of once
# per hyperparameter set: it is about half the work of a 'compute_metrics' job, which the grid
# would otherwise repeat identically a few thousand times.
rule compute_baseline_metrics:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        swd_distributions=expand(swd_distribution_file("{param_set}"),
                                 param_set=PARAM_SETS, allow_missing=True),
        target_topology_rates_file=TARGET_TOPOLOGY_RATES_FILE

    params:
        sample_dir=BASE_SAMPLES_DIR,
        binning_file=config["parameters"]["binning_file"],
        n_directions=config["swd_bootstrapping"]["n_directions"],
        param_set_dict=json.dumps(PARAM_SET_DICT),
        swd_seed=lambda wc: swd_direction_seed(wc.sample, wc.topology),
        subsample_seed=lambda wc: data_subsample_seed(wc.sample, wc.topology),
        n_subsamples=N_SUBSAMPLES,
        data_n_events=data_n_events_for,
        swd_distribution_flags=named_paths_flags("--swd_distribution", swd_distributions_of(PARAM_SETS))

    output:
        baseline_metrics_file=BASELINE_METRICS_FILE

    threads: 1
    # This job loads no model, so it sits at the sample-loading floor of 'compute_metrics' (3.0 GB
    # for CC0pi). It is given the same wall time as a 'compute_metrics' job despite taking a little
    # under that of one (98 minutes for CC0pi): it blocks every hyperparameter set of its topology,
    # so it is not the place to be tight.
    resources:
        mem_mb=baseline_metrics_mem_mb,
        runtime=metrics_runtime,

    benchmark:
        BENCH_DIR + "compute_baseline_metrics.tsv"

    conda:
        ENV

    shell:
        """
        python Compute_baseline_metrics.py \
            --sample_dir {params.sample_dir} \
            --topology {wildcards.topology} \
            {params.swd_distribution_flags} \
            --n_directions {params.n_directions} \
            --param_set_dict '{params.param_set_dict}' \
            --binning_file {params.binning_file} \
            --swd_seed {params.swd_seed} \
            --subsample_seed {params.subsample_seed} \
            --n_subsamples {params.n_subsamples} \
            --data_n_events {params.data_n_events} \
            --target_topology_rates_file {input.target_topology_rates_file} \
            --output_file {output.baseline_metrics_file}
        """

# Computes the relevant metrics for a single trained model, and saves them to a .csv file. 
# This is run once for each trained model for each sample/topology combination.
rule compute_metrics:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        model_dir = MODEL_DIR,
        baseline_metrics=BASELINE_METRICS_FILE,
        swd_distributions=expand(swd_distribution_file("{param_set}"),
                                 param_set=PARAM_SETS, allow_missing=True),
        target_topology_rates_file=TARGET_TOPOLOGY_RATES_FILE

    params:
        sample_dir=BASE_SAMPLES_DIR,
        model=lambda wc: wc.model,
        logdir=lambda wc: TENSORBOARD_RUN_DIR.format(sample=wc.sample, topology=wc.topology,
                                                     model=wc.model, hp_id=wc.hp_id),
        binning_file=config["parameters"]["binning_file"],
        n_directions=config["swd_bootstrapping"]["n_directions"],
        param_set_dict=json.dumps(PARAM_SET_DICT),
        swd_seed=lambda wc: swd_direction_seed(wc.sample, wc.topology),
        subsample_seed=lambda wc: data_subsample_seed(wc.sample, wc.topology),
        n_subsamples=N_SUBSAMPLES,
        data_n_events=data_n_events_for,
        swd_distribution_flags=named_paths_flags("--swd_distribution", swd_distributions_of(PARAM_SETS))

    output:
        metrics_file=METRICS_DIR + "metrics.csv",
        tensorboard_dir=directory(TENSORBOARD_RUN_DIR)

    # group: "tune"
    threads: 2

    # See 'metrics_runtime' and 'metrics_mem_mb'. The data-scale subsamples add little, as they are
    # a few percent of the test sample each.
    resources:
        mem_mb=metrics_mem_mb,
        runtime=metrics_runtime,
        # disk_mb=10000

    benchmark:
        BENCH_DIR + "compute_metrics_{model}/compute_metrics_{model}_{hp_id}.tsv"

    conda:
        ENV

    shell:
        """
        python Compute_metrics.py \
            --sample_dir {params.sample_dir} \
            --model {params.model} \
            --model_dir {input.model_dir} \
            --topology {wildcards.topology} \
            {params.swd_distribution_flags} \
            --logdir {params.logdir} \
            --n_directions {params.n_directions} \
            --param_set_dict '{params.param_set_dict}' \
            --binning_file {params.binning_file} \
            --baseline_metrics {input.baseline_metrics} \
            --swd_seed {params.swd_seed} \
            --subsample_seed {params.subsample_seed} \
            --n_subsamples {params.n_subsamples} \
            --data_n_events {params.data_n_events} \
            --target_topology_rates_file {input.target_topology_rates_file} \
            --output_file {output.metrics_file}
        """

# Local rule that finds the best-performing model for every model in a given sample/topology,
# according to the metric specified in the config. The hyperparameters and metrics for each 
# model are saved in a single .json file. A symlink is created to each of the best-performing 
# models in saved_models/{tag}/{sample}/{topology}/{model}. This rule is run once for each 
# sample/topology combination.
rule choose_models:
    input:
        lambda wc: [
            METRICS_DIR.format(sample=wc.sample, topology=wc.topology, model=model, hp_id=hp_id)
            + f"metrics.csv"
            for model in MODELS
            for hp_id in hp_labels_of(wc.sample, wc.topology, model)
        ]
    output:
        metrics_file=f"saved_metrics/{TAG}/{{sample}}/{{topology}}/metrics.json",
        model_files=[model_file(model) for model in MODELS]

    params:
        input_dir = ALL_METRICS_DIR,
        selections=" ".join(
            f"--selection {model} {SELECTION_METRIC[model]['metric']} {SELECTION_METRIC[model]['direction']}"
            for model in MODELS
        ),
        grid_file=GRID_FILE,
        excluded=lambda wc: " ".join(
            f"--exclude {model} {hp_id}"
            for model in MODELS
            for hp_id in set(HP_LABELS[model]) - set(hp_labels_of(wc.sample, wc.topology, model))
        ),
        model_dir=ALL_MODELS_DIR,
        final_model_dir = FINAL_MODEL_DIR

    conda:
        ENV
    shell:
        """
        python Gather_metrics.py \
            --input_dir {params.input_dir} \
            --metrics_file {output.metrics_file} \
            --model_dir {params.model_dir} \
            --final_model_dir {params.final_model_dir} \
            --grid_file {params.grid_file} \
            {params.selections} {params.excluded}
        """

# Generates plots showing the reweighting performance of each model for every parameter included
# in the config, and the triangle plots of the reweighting parameters (the original and target
# distributions, and the target over the reweighted distribution and their pull). For BDT-based
# reweighting models, plots of the training history are also generated. This rule is run once for
# each sample/topology combination, and the plots are saved as .pdf files.
rule make_plots:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        model_files=[model_file(model) for model in MODELS],
        target_topology_rates_file=TARGET_TOPOLOGY_RATES_FILE

    output:
        histograms_1D=FIG_DIR + "1Dhist.pdf",
        histograms_1D_data=FIG_DIR + "1Dhist_data.pdf",
        triangles=[FIG_DIR + f"triangle_{name}.pdf" for name in TRIANGLE_PLOTS]

    params:
        sample_dir=BASE_SAMPLES_DIR,
        binning_file=config["parameters"]["binning_file"],
        output_dir=FIG_DIR,
        model_dir=FINAL_MODEL_DIR,
        model_list=MODELS,
        reweight_params=PARAM_SET_DICT[REWEIGHTING_SET],
        data_n_events=data_n_events_for,
        subsample_seed=lambda wc: data_subsample_seed(wc.sample, wc.topology)

    threads: 1
    # The depth-9 XGB models run to several hundred MB of json (407 MB for CC1pipm), and loading
    # them took make_plots over 4 GB, against ~1.5 GB for the others. The grid only grants more
    # than 4 GB to jobs under 3 hours, so the wall time is a flat 180 minutes rather than one grown
    # by the retry attempt; the slowest make_plots job so far took ~5 minutes.
    resources:
        mem_mb=12000,
        runtime=180,
        # disk_mb=10000

    benchmark:
        BENCH_DIR + "make_plots.tsv"

    conda:
        ENV

    shell:
        """
        python Make_plots.py \
            --sample_dir {params.sample_dir} \
            --topology {wildcards.topology} \
            --model_dir {params.model_dir} \
            --model_list {params.model_list} \
            --reweight_params {params.reweight_params} \
            --output_dir {params.output_dir} \
            --make_1D_plots \
            --make_triangle_plots \
            --binning_file {params.binning_file} \
            --data_n_events {params.data_n_events} \
            --subsample_seed {params.subsample_seed} \
            --target_topology_rates_file {input.target_topology_rates_file}
        """

# Local rule that plots, for every model and the original sample before reweighting, a heatmap of the
# data-scale SWD p-value of every sample/topology combination on each parameter set, once the
# metrics of every combination are gathered. It spans the whole tag, so it is run once, rather than
# by make_plots for every combination; it only reads the metrics.json files, and takes seconds.
rule plot_pvalue_heatmaps:
    input:
        metrics_files=all_metrics_files

    output:
        heatmaps=HEATMAP_FILES

    params:
        metrics_dir=f"saved_metrics/{TAG}/",
        output_dir=HEATMAP_DIR,
        models=HEATMAP_MODELS,
        param_sets=PARAM_SETS

    conda:
        ENV

    shell:
        """
        python Plot_pvalue_heatmap.py \
            --metrics_dir {params.metrics_dir} \
            --metrics_files {input.metrics_files} \
            --output_dir {params.output_dir} \
            --models {params.models} \
            --param_sets {params.param_sets}
        """

# Local rule that records the configuration a tag was run with: the config as Snakemake resolved
# it (config.yaml with every --configfile and --config override merged over it, e.g. the
# config_scarf.yaml of the SCARF profile) and a copy of the hyperparameter grid file. The config is
# a param, so a later run of the same tag with a different config rewrites the record.
rule save_configs:
    input:
        grid_file=GRID_FILE
    output:
        config_file=CONFIGS_DIR + "config.yaml",
        grid_file=CONFIGS_DIR + os.path.basename(GRID_FILE)
    params:
        config=config
    run:
        import shutil
        import yaml
        with open(output.config_file, "w") as f:
            yaml.safe_dump(params.config, f, sort_keys=False)
        shutil.copyfile(input.grid_file, output.grid_file)

# Carries out the entire workflow, by requesting the outputs of the initialize_analysis
# checkpoint for every sample, the outputs of choose_model and make_plots for every
# sample/topology combination, and the p-value heatmaps of the tag. This rule is run once during
# the workflow.
rule all:
    input:
        included_topologies_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES),
        metrics_files=all_metrics_files,
        figures=all_figure_dirs,
        heatmaps=HEATMAP_FILES,
        configs=rules.save_configs.output

# Carries out the intialize_analysis checkpoint for every sample by requesting its output.
# This rule can be run before a dry run to see the full DAG, since the number of topologies
# passing the event count check is only known after the checkpoint has run.
rule count_all_topologies:
    input:
        counts_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES)

