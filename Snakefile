configfile: "config.yaml"

import glob
import json
import os
import re

from List_hyperparameters import number_of_sets
from Split_sizes import minimum_events
from Param_sets import ALL_PARAMS_SET, REWEIGHTING_SET

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
RUNS = range(config["swd_bootstrapping"]["runs"])

# ---------------------------------------------------------------------------
# Sets of analysis parameters
# The metrics are computed for every set listed in the config file, plus the 'all' set holding
# every analysis parameter and the set of parameters the models are reweighted on. Every set is
# handled by the same rules, with a {param_set} wildcard naming it, so that adding a set to the
# config file is enough to have its samples, bootstrapped SWD distribution and metrics produced.
# ---------------------------------------------------------------------------

METRIC_SETS = dict(config["parameters"].get("metric_sets") or {})
# Sets written by the initialisation, i.e. every set but the reweighting one, whose samples are
# obtained by keeping the reweighting parameters of the 'all' samples.
INIT_SETS = [ALL_PARAMS_SET] + list(METRIC_SETS)
PARAM_SETS = INIT_SETS + [REWEIGHTING_SET]

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

def get_param_set_dict():
    dict = METRIC_SETS.copy()
    dict[ALL_PARAMS_SET] = config["parameters"]["all"]
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
INIT_SAMPLES_DIR = f"saved_samples/{TAG}/{{sample}}/{{topology}}/"
INIT_SWD_DIR     = f"saved_swd_distribution/{TAG}/{{sample}}/{{topology}}/"
SAMPLES_DIR      = INIT_SAMPLES_DIR + f"{REWEIGHTING_SET}/"
SWD_DIR          = INIT_SWD_DIR + f"{REWEIGHTING_SET}/"
MODEL_DIR        = f"saved_models/{TAG}/{{sample}}/{{topology}}/{REWEIGHTING_SET}/"
WEIGHTS_DIR      = f"saved_weights/{TAG}/{{sample}}/{{topology}}/{REWEIGHTING_SET}/"
FIG_DIR          = f"saved_figures/{TAG}/{{sample}}/{{topology}}/{REWEIGHTING_SET}/"
METRICS_DIR      = f"saved_metrics/{TAG}/{{sample}}/{{topology}}/{REWEIGHTING_SET}/"
HPS_DIR          = f"hps/{TAG}/"
TENSORBOARD_DIR  = f"TensorBoard/{TAG}/{{sample}}/{{topology}}/{REWEIGHTING_SET}/"

# The bootstrapped SWD distribution of a set of parameters, whichever rule wrote its samples.
def swd_distribution_file(param_set):
    return INIT_SWD_DIR + f"{param_set}/swd_distribution_{param_set}.npy"

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

# ---------------------------------------------------------------------------
# Topology filtering
# A topology holding too few events cannot be split into training, validation
# and test samples, so it is skipped for the sample it is too sparse in.
# The minimum depends on the configured split percentages: with the default
# 40%/40% split, 3 events are needed (2 events would give int(0.4*2) = 0
# training and 0 validation events).
# The same topology is still trained on in the samples where it is filled.
# ---------------------------------------------------------------------------

MIN_EVENTS_PER_TOPOLOGY = minimum_events(
    config["analysis"]["train_percentage"], config["analysis"]["val_percentage"]
)
# The counts of all topologies are gathered in a single per-sample file (no {topology} wildcard),
# as they are all obtained from one pass over the ROOT files of the sample.
INCLUDED_TOPOLOGIES_FILE = f"saved_samples/{TAG}/{{sample}}/included_topologies.json"

def topologies_for_sample(sample):
    """Return the topologies of the config file that hold enough events in a given sample."""
    with open(checkpoints.initialize_analysis.get(sample=sample).output.included_topologies) as f:
        included_topologies = json.load(f)
    
    return included_topologies

def all_metrics_files(wildcards):
    return [
        f"saved_metrics/{TAG}/{sample}/{topology}/{REWEIGHTING_SET}/metrics.json"
        for sample in SAMPLES
        for topology in topologies_for_sample(sample)
    ]

def all_figure_dirs(wildcards):
    return [
        f"saved_figures/{TAG}/{sample}/{topology}/{REWEIGHTING_SET}/1Dhist.pdf"
        for sample in SAMPLES
        for topology in topologies_for_sample(sample)
    ]

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
        samples_dir=BASE_SAMPLES_DIR

    output:
        last_sampled_files=directory(BASE_SAMPLES_DIR + "target_test.parquet"),
        included_topologies=INCLUDED_TOPOLOGIES_FILE

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
            --output_dir {params.samples_dir} \
            --topologies_file {output.included_topologies} \
        """

# The bootstrapped SWD distribution of every set of analysis parameters is obtained the same way,
# whether its samples were written by the initialisation or by the reweighting split: a single
# pair of rules covers them all, with the set named by the {param_set} wildcard.
rule run_bootstrap:
    input:
        target_test=BASE_SAMPLES_DIR + "target_test.parquet"
    output:
        output_file=INIT_SWD_DIR + "{param_set}/indiv_bootstrap/run_{run_id}.npy"
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"],
        param_set=lambda wc: PARAM_SET_DICT[wc.param_set],
        n_bootstrap=config["swd_bootstrapping"]["n_samples"]
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

rule aggregate_bootstrap:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/{wc.topology}/{wc.param_set}/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=INIT_SWD_DIR + "{param_set}/swd_distribution_{param_set}.npy"
    conda:
        ENV
    shell:
        """
        python Aggregate.py \
            --input {input} \
            --output {output.final_file}
        """


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

rule single_fine_tuning:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        hparam_file=HPS_DIR + "{model}/{model}_hp_{run_id}.json",
        swd_distributions=expand(INIT_SWD_DIR + "{param_set}/swd_distribution_{param_set}.npy",
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
        output_file=TENSORBOARD_DIR + "{model}/run_{run_id}_metrics.csv"

    conda:
        ENV

    shell:
        """
        python Fine_tuning.py \
            --sample_dir {params.sample_dir} \
            --model {params.model} \
            --topology {wildcards.topology} \
            {params.swd_distribution_flags} \
            --hyperparameters {input.hparam_file} \
            --logdir {params.logdir} \
            --n_directions {params.n_directions} \
            --param_set_dict '{params.param_set_dict}' \
            --binning_file {params.binning_file} \
            --output_file {output.output_file}
        """

rule fine_tuning:
    input:
        lambda wc: [
            f"TensorBoard/{TAG}/{wc.sample}/{wc.topology}/{REWEIGHTING_SET}/{model}/run_{run_id}_metrics.csv"
            for model in MODELS
            for run_id in range(NUMBER_OF_SETS[model])
        ]
    output:
        output_file=f"set_hyperparameters/{TAG}/{{sample}}/{{topology}}/hyperparameters.json"
    params:
        logdir=TENSORBOARD_DIR,
        grid_file=GRID_FILE,
        # One '--selection MODEL METRIC DIRECTION' argument per model.
        selections=" ".join(
            f"--selection {model} {SELECTION_METRIC[model]['metric']} {SELECTION_METRIC[model]['direction']}"
            for model in MODELS
        )
    conda:
        ENV
    shell:
        """
        python Gather_fine_tuning.py \
            --input_dir {params.logdir} \
            --output_file {output.output_file} \
            --grid_file {params.grid_file} \
            {params.selections}
        """


rule train_models:
    input:
        hparam_file=f"set_hyperparameters/{TAG}/{{sample}}/{{topology}}/hyperparameters.json",
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet"
    params:
        samples_dir=BASE_SAMPLES_DIR,
        model_list=MODELS,
        model_dir=MODEL_DIR,
        weights_dir=WEIGHTS_DIR,
        param_set=PARAM_SET_DICT[REWEIGHTING_SET]
    output:
        save_path_dict=WEIGHTS_DIR + f"weights_path_dict_{REWEIGHTING_SET}.json"
    conda:
        ENV
    shell:
        """
        python Training.py \
            --sample_dir {params.samples_dir} \
            --topology {wildcards.topology} \
            --param_set {params.param_set} \
            --hparams_dict {input.hparam_file} \
            --save_weights_path {params.weights_dir} \
            --save_model_path {params.model_dir} \
            --model_list {params.model_list} \
            --save_path_dict {output.save_path_dict}
        """


rule compute_metrics:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        weights_path=WEIGHTS_DIR + f"weights_path_dict_{REWEIGHTING_SET}.json",
        swd_distributions=expand(INIT_SWD_DIR + "{param_set}/swd_distribution_{param_set}.npy",
                                 param_set=PARAM_SETS, allow_missing=True)

    output:
        metrics_file=METRICS_DIR + "metrics.json"

    params:
        binning_file=config["parameters"]["binning_file"],
        sample_dir=BASE_SAMPLES_DIR,
        swd_distribution_flags=named_paths_flags("--swd_distribution", swd_distributions_of(PARAM_SETS)),
        n_directions=config["swd_bootstrapping"]["n_directions"],
        param_set_dict=json.dumps(PARAM_SET_DICT)

    conda:
        ENV

    shell:
        """
        python Calc_Metrics.py \
            --sample_dir {params.sample_dir} \
            {params.swd_distribution_flags} \
            --param_set_dict '{params.param_set_dict}' \
            --topology {wildcards.topology} \
            --weights_paths {input.weights_path} \
            --output_file {output.metrics_file} \
            --compute_chi2 \
            --compute_swd \
            --n_directions {params.n_directions} \
            --binning_file {params.binning_file}
        """

rule make_plots:
    input:
        last_sampled_file=BASE_SAMPLES_DIR + "target_test.parquet",
        weights_path=WEIGHTS_DIR + f"weights_path_dict_{REWEIGHTING_SET}.json"

    output:
        histograms_1D=FIG_DIR + "1Dhist.pdf"

    params:
        sample_dir=BASE_SAMPLES_DIR,
        binning_file=config["parameters"]["binning_file"],
        output_dir=FIG_DIR

    conda:
        ENV

    shell:
        """
        python Make_plots.py \
            --sample_dir {params.sample_dir} \
            --topology {wildcards.topology} \
            --weights_paths {input.weights_path} \
            --output_dir {params.output_dir} \
            --make_1D_plots \
            --no-make_2D_plots \
            --binning_file {params.binning_file}
        """

rule all:
    input:
        # The counts are requested explicitly so that every sample is scanned in one go,
        # rather than one sample at a time as the checkpoints get resolved.
        included_topologies_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES),
        metrics_files=all_metrics_files,
        figures=all_figure_dirs

# Wildcard-free target running only the topology counting checkpoint on every sample.
# The 'initialize_analysis' checkpoint itself carries a {sample} wildcard and so cannot be
# asked for on the command line; ask for this rule instead:
#     snakemake count_all_topologies --cores 8
# Running it first makes the counts available to the checkpoints, so that a subsequent
# dry run ('snakemake all -n') can resolve them and list every job of the workflow.
rule count_all_topologies:
    input:
        counts_files=expand(INCLUDED_TOPOLOGIES_FILE, sample=SAMPLES)

