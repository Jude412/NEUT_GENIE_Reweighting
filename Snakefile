configfile: "config.yaml"

from itertools import chain
import glob
import hashlib
import json
import os
import re

from Constants import DATA_N_EVENTS, DEFAULT_DATA_N_EVENTS
from List_hyperparameters import labels_of
from Param_sets import ALL_PARAMS_SET, REWEIGHTING_SET
from Train_predict import model_extension

ENV = "environment.yaml"
TAG = config["output"]["tag"]
MODELS = config["models"]["model_list"]
GRID_FILE = config["models"]["grid_file"]
HP_LABELS = {model: labels_of(GRID_FILE, model) for model in MODELS}
SELECTION_METRIC = config["models"]["selection_metric"]
MODES = config["analysis"]["modes"]
TOPOLOGIES = config["analysis"]["topologies"]
# Optional fraction of each input file's events to read in; unset (None) reads every event.
DOWNSAMPLING = config["analysis"].get("downsampling")
# Optional maximum number of events to be used for training; unset (None) trains on every event.
MAX_TRAINING_EVENTS = config["analysis"].get("max_training_events")

RUNS = range(config["swd_bootstrapping"]["runs"])
# Number of data-sized subsamples the data-scale SWD and p-value of every run are averaged over.
N_SUBSAMPLES = config["data_scale"]["n_subsamples"]

def data_n_events_for(wildcards):
    """Return the total number of raw events the real downstream analysis will use for a sample."""
    return DATA_N_EVENTS.get(wildcards.sample, DEFAULT_DATA_N_EVENTS)

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

# Wall time (minutes) a 'train_model' job is given, per topology. It scales with the number of
# events the topology holds, which the training cap only partly levels out: at
# 'max_training_events' of 2.5M, CC0pi loads ~9.4M events against ~1.1M for CCOther. The values
# are the worst-case run-3 benchmarks (benchmarks/{tag}/{sample}/{topology}/train_model_XGB/,
# ~25 minutes per million events loaded) scaled to the capped sample size, with margin
#
# full_params2 doubled 'n_estimators' to 10000. Extrapolating the full_params benchmarks linearly in
# the number of trees, the worst cases at 10000 trees are CC0pi 485, CC1pipm 466, NC0pi 444 and
# CCNpi 273 minutes. NC0pi already went over 180 minutes on 129 of its 300 full_params runs, which
# only finished through the retry.
TRAIN_RUNTIME = {
    "CC0pi":   600,
    "CC1pipm": 600,
    "NC0pi":   600,
    "CCNpi":   360,
}
DEFAULT_TRAIN_RUNTIME = 180

# The grid only grants more than 4 GB to jobs that run for less than 3 hours, and the topologies
# above all run for longer, so the memory request is a flat 4 GB rather than a per-topology one
# that would leave them unmatched. It is not grown by the retry attempt either, for the same
# reason: a retry asking for 8 GB over 3 hours would never be scheduled. The training cap is what
# keeps the peak inside that ceiling (~2.9 GB for CC0pi, the largest, against a measured
# 96 MB + 298 MB per million events loaded), so raising the cap means re-checking this.
TRAIN_MEM_MB = 4000

def train_model_runtime(wildcards, attempt):
    """Return the wall time a train_model job is given, grown by the retry attempt as 'scaled'
    does for the other rules."""
    return TRAIN_RUNTIME.get(wildcards.topology, DEFAULT_TRAIN_RUNTIME) * attempt


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

def all_figure_dirs(wildcards):
    return [
        f"saved_figures/{TAG}/{sample}/{topology}/{name}"
        for sample in SAMPLES
        for topology in topologies_for_sample(sample)
        for name in ("1Dhist.pdf", "1Dhist_data.pdf")
    ]

localrules: prepare_hps, aggregate_bootstrap, choose_models

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
    resources:
        mem_mb=12000,
        runtime=180,
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
        runtime=scaled(180),
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
        hparam_file=HPS_DIR + "{model}/{model}_hp_{hp_id}.json"
    params:
        sample_dir=BASE_SAMPLES_DIR,
        model=lambda wc: wc.model,
        reweight_params=PARAM_SET_DICT[REWEIGHTING_SET],
        max_events=MAX_TRAINING_EVENTS,
        seed=lambda wc: training_seed(wc.sample, wc.topology)
    output:
        model_dir=directory(MODEL_DIR)
    threads: 1
    resources:
        mem_mb=TRAIN_MEM_MB,
        runtime=train_model_runtime,
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
    # This job loads no model, so it sits at the sample-loading floor the 'compute_metrics'
    # benchmarks measure (~2.8 GB). It is given the same wall time as a 'compute_metrics' job
    # despite being expected to take about half of one: it blocks every hyperparameter set of its
    # topology, so it is not the place to be tight.
    resources:
        mem_mb=4000,
        runtime=scaled(180),

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
    threads: 1

    # 25 of the 3000 full_params jobs already peaked above 4 GB (up to 4.5 GB, CC0pi), and the
    # depth-9 models at 10000 trees extrapolate to 4.2-4.8 GB. The grid only grants more than 4 GB
    # to jobs under 3 hours, so the wall time is a flat 175 minutes rather than one grown by the
    # retry attempt. The longest full_params job took 95 minutes (CC0pi); a linear fit in the number
    # of trees puts CC0pi at ~75 minutes at 10000 trees, so that slowest job looks like an outlier
    # rather than the trend.
    resources:
        mem_mb=6000,
        runtime=175,
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
            for hp_id in HP_LABELS[model]
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
            {params.selections}
        """

# Generates plots showing the reweighting performance of each model for every parameter included
# in the config. For BDT-based reweighting models, plots of the training history are also 
# generated. This rule is run once for each sample/topology combination, and the plots are saved 
# as .pdf files.
rule make_plots:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        model_files=[model_file(model) for model in MODELS],
        target_topology_rates_file=TARGET_TOPOLOGY_RATES_FILE

    output:
        histograms_1D=FIG_DIR + "1Dhist.pdf",
        histograms_1D_data=FIG_DIR + "1Dhist_data.pdf"

    params:
        sample_dir=BASE_SAMPLES_DIR,
        binning_file=config["parameters"]["binning_file"],
        output_dir=FIG_DIR,
        model_dir=FINAL_MODEL_DIR,
        model_list=MODELS,
        reweight_params=PARAM_SET_DICT[REWEIGHTING_SET],
        data_n_events=data_n_events_for

    threads: 1
    resources:
        mem_mb=4000,
        runtime=scaled(180),
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
            --no-make_2D_plots \
            --binning_file {params.binning_file} \
            --data_n_events {params.data_n_events} \
            --target_topology_rates_file {input.target_topology_rates_file}
        """

# Carries out the entire workflow, by requesting the outputs of the initialize_analysis
# checkpoint for every sample, and the outputs of choose_model and make_plots for every
# sample/topology combination. This rule is run once during the workflow.
rule all:
    input:
        included_topologies_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES),
        metrics_files=all_metrics_files,
        figures=all_figure_dirs

# Carries out the intialize_analysis checkpoint for every sample by requesting its output.
# This rule can be run before a dry run to see the full DAG, since the number of topologies
# passing the event count check is only known after the checkpoint has run.
rule count_all_topologies:
    input:
        counts_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES)

