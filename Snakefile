configfile: "config.yaml"

from itertools import chain
import glob
import json
import os
import re

from List_hyperparameters import number_of_sets
from Split_sizes import minimum_events
from Param_sets import ALL_PARAMS_SET, REWEIGHTING_SET
from Train_predict import model_extension

ENV = "environment.yaml"
TAG = config["output"]["tag"]
MODELS = config["models"]["model_list"]
GRID_FILE = config["models"]["grid_file"]
# The number of fine-tuning runs of a model is the size of its hyperparameter grid, so that the
# grid is only ever described in the grid file.
NUMBER_OF_SETS = {model: number_of_sets(GRID_FILE, model) for model in MODELS}
SELECTION_METRIC = config["models"]["selection_metric"]
MODES = config["analysis"]["modes"]
TOPOLOGIES = config["analysis"]["topologies"]
# Optional fraction of each input file's events to read in; unset (None) reads every event.
DOWNSAMPLING = config["analysis"].get("downsampling")
RUNS = range(config["swd_bootstrapping"]["runs"])

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
MODEL_DIR        = BASE_MODEL_DIR + f"{{run_id}}/"
FINAL_MODEL_DIR  = f"saved_final_models/{TAG}/{{sample}}/{{topology}}/"
ALL_METRICS_DIR  = f"saved_metrics/{TAG}/{{sample}}/{{topology}}/"
BASE_METRICS_DIR = f"saved_metrics/{TAG}/{{sample}}/{{topology}}/{{model}}/"
METRICS_DIR      = BASE_METRICS_DIR + f"{{run_id}}/"
FIG_DIR          = f"saved_figures/{TAG}/{{sample}}/{{topology}}/"
HPS_DIR          = f"hps/{TAG}/"
TENSORBOARD_DIR  = f"TensorBoard/{TAG}/{{sample}}/{{topology}}/"

# Benchmarks hold the peak memory and wall time of every submitted job, and are what the
# resource requests of the rules are calibrated from. They mirror the wildcards of the rule
# that wrote them.
BASE_BENCH_DIR   = f"benchmarks/{TAG}/{{sample}}/"
BENCH_DIR   = BASE_BENCH_DIR + "{topology}/"

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
    for run_id in range(NUMBER_OF_SETS[model]):
        HPS_FILES.append(os.path.join(HPS_DIR, f"{model}/{model}_hp_{run_id}.json"))

# Wildcard constraint: sample paths contain only word characters and slashes.
# The topology wildcard is restricted to the topology names listed in the config file,
# which also removes the ambiguity with the (slash-containing) sample wildcard.
# The parameter set wildcard is likewise restricted to the configured sets.
wildcard_constraints:
    sample="[^.]+",
    topology="|".join(re.escape(str(t)) for t in TOPOLOGIES),
    param_set="|".join(re.escape(str(s)) for s in PARAM_SETS),
    run_id=r"\d+"

INCLUDED_TOPOLOGIES_FILE = f"saved_samples/{TAG}/{{sample}}/included_topologies.json"

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
        f"saved_figures/{TAG}/{sample}/{topology}/1Dhist.pdf"
        for sample in SAMPLES
        for topology in topologies_for_sample(sample)
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
        included_topologies=INCLUDED_TOPOLOGIES_FILE

    threads: 1
    resources:
        mem_mb=4000,
        runtime=scaled(180),
        disk_mb=10000

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
        """

# Bootstraps swd_bootstrapping/n_samples samples from the target test sample, calculates 
# the SWD between each sample and the target test sample, and saves the SWD distribution 
# as a .npy file. This rule is run swd_bootstrapping/runs times for each sample/topology/
# parameter set combination. These are marked as temp, since they are later aggregated into 
# a single file.
rule run_bootstrap:
    input:
        target_test=BASE_SAMPLES_DIR + "target_test.parquet"
    output:
        output_file=temp(INIT_SWD_DIR + "{param_set}/indiv_bootstrap/run_{run_id}.npy")
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"],
        param_set=lambda wc: PARAM_SET_DICT[wc.param_set],
        n_bootstrap=config["swd_bootstrapping"]["n_samples"]
    # group: "swd"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=scaled(180),
        disk_mb=10000
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
        hparam_file=HPS_DIR + "{model}/{model}_hp_{run_id}.json"
    params:
        sample_dir=BASE_SAMPLES_DIR,
        model=lambda wc: wc.model,
        reweight_params=PARAM_SET_DICT[REWEIGHTING_SET]
    output:
        model_dir=directory(MODEL_DIR)
    threads: 1
    resources:
        mem_mb=4000,
        runtime=scaled(500),
        disk_mb=10000
    benchmark:
        BENCH_DIR + f"train_model_{{model}}/train_model_{{model}}_hp_{{run_id}}.tsv"
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
            --output_dir {output.model_dir}
        """

# Computes the relevant metrics for a single trained model, and saves them to a .csv file. 
# This is run once for each trained model for each sample/topology combination.
rule compute_metrics:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        model_dir = MODEL_DIR,
        hparam_file=HPS_DIR + "{model}/{model}_hp_{run_id}.json",
        swd_distributions=expand(swd_distribution_file("{param_set}"),
                                 param_set=PARAM_SETS, allow_missing=True)

    params:
        sample_dir=BASE_SAMPLES_DIR,
        model=lambda wc: wc.model,
        logdir=lambda wc: TENSORBOARD_DIR + wc.model + "/",
        binning_file=config["parameters"]["binning_file"],
        n_directions=config["swd_bootstrapping"]["n_directions"],
        param_set_dict=json.dumps(PARAM_SET_DICT),
        swd_distribution_flags=named_paths_flags("--swd_distribution", swd_distributions_of(PARAM_SETS))

    output:
        metrics_file=METRICS_DIR + "metrics.csv"

    # group: "tune"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=scaled(180),
        disk_mb=10000

    benchmark:
        BENCH_DIR + "compute_metrics_{model}/compute_metrics_{model}_{run_id}.tsv"

    conda:
        ENV

    shell:
        """
        python Compute_metrics.py \
            --sample_dir {params.sample_dir} \
            --model {params.model} \
            --model_dir {input.model_dir} \
            --topology {wildcards.topology} \
            --hyperparameters {input.hparam_file} \
            {params.swd_distribution_flags} \
            --logdir {params.logdir} \
            --n_directions {params.n_directions} \
            --param_set_dict '{params.param_set_dict}' \
            --binning_file {params.binning_file} \
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
            METRICS_DIR.format(sample=wc.sample, topology=wc.topology, model=model, run_id=run_id)
            + f"metrics.csv"
            for model in MODELS
            for run_id in range(NUMBER_OF_SETS[model])
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
        model_files=[model_file(model) for model in MODELS]

    output:
        histograms_1D=FIG_DIR + "1Dhist.pdf"

    params:
        sample_dir=BASE_SAMPLES_DIR,
        binning_file=config["parameters"]["binning_file"],
        output_dir=FIG_DIR,
        model_dir=ALL_MODELS_DIR,
        model_list=MODELS,
        reweight_params=PARAM_SET_DICT[REWEIGHTING_SET]

    threads: 1
    resources:
        mem_mb=4000,
        runtime=scaled(180),
        disk_mb=10000

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
            --binning_file {params.binning_file}
        """

# Carries out the entire workflow, by requesting the outputs of the initialize_analysis
# checkpoint for every sample, and the outputs of choose_model and make_plots for every
# sample/topology combination. This rule is run once during the workflow.
rule all:
    input:
        # The counts are requested explicitly so that every sample is scanned in one go,
        # rather than one sample at a time as the checkpoints get resolved.
        included_topologies_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES),
        metrics_files=all_metrics_files,
        figures=all_figure_dirs

# Carries out the intialize_analysis checkpoint for every sample by requesting its output.
# This rule can be run before a dry run to see the full DAG, since the number of topologies
# passing the event count check is only known after the checkpoint has run.
rule count_all_topologies:
    input:
        counts_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES)

