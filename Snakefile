configfile: "config.yaml"

import glob
import json
import os
import re

from List_hyperparameters import number_of_sets
from Split_sizes import minimum_events

ENV = "environment.yaml"
TAG = config["output"]["tag"]
DIM = len(config["parameters"]["reweighting"])
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

ALL_PARAMS_SET = "all"
REWEIGHTING_SET = f"custom_{DIM}D"
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
# computing the metrics take them ('--sample_dir 3D path/to/3D/ --sample_dir 8D path/to/8D/ ...').
def named_paths_flags(flag, paths):
    return " ".join(f"{flag} {param_set} {path}" for param_set, path in paths.items())

def sample_dirs_of(param_sets):
    return {param_set: INIT_SAMPLES_DIR + f"{param_set}/" for param_set in param_sets}

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
TOPOLOGY_COUNTS_FILE = f"saved_samples/{TAG}/{{sample}}/topology_counts.json"

def topologies_for_sample(sample):
    """Return the topologies of the config file that hold enough events in a given sample."""
    with open(checkpoints.count_topologies.get(sample=sample).output.counts_file) as f:
        counts = json.load(f)

    kept = []
    for topology in TOPOLOGIES:
        n_original = counts["original"][str(topology)]
        n_target = counts["target"][str(topology)]
        if n_original < MIN_EVENTS_PER_TOPOLOGY or n_target < MIN_EVENTS_PER_TOPOLOGY:
            print(
                f"Warning: topology {topology} of sample {sample} holds {n_original} original event(s) "
                f"and {n_target} target event(s). Topologies with less than {MIN_EVENTS_PER_TOPOLOGY} "
                "events in either the original or the target sample cannot be split into training, "
                "validation and test samples: no training will be carried out for this topology in this sample."
            )
            continue
        kept.append(topology)

    return kept

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

checkpoint count_topologies:
    input:
        original_file=original_file_for,
        target_file=target_file_for

    params:
        modes=MODES,
        topologies=TOPOLOGIES,
        original_tree=config["inputs"]["original_tree"],
        target_tree=config["inputs"]["target_tree"],
        branches=config["inputs"]["branches"]

    output:
        counts_file=TOPOLOGY_COUNTS_FILE

    conda:
        ENV

    shell:
        """
        python Count_topologies.py \
            --input_file_original {input.original_file} \
            --input_file_target {input.target_file} \
            --input_tree_original {params.original_tree} \
            --input_tree_target {params.target_tree} \
            --branches {params.branches} \
            --modes {params.modes} \
            --topologies {params.topologies} \
            --output_file {output.counts_file}
        """


rule initialize_analysis:
    input:
        original_file=original_file_for,
        target_file=target_file_for

    params:
        modes=MODES,
        original_tree=config["inputs"]["original_tree"],
        target_tree=config["inputs"]["target_tree"],
        # neutrino_PDG=config["analysis"]["neutrino_PDG"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"],
        branches=config["inputs"]["branches"],
        analysis_params=config["parameters"]["all"],
        # One '--param_set NAME param1 param2 ...' argument per configured set of parameters.
        # The 'all' set is always written by the script itself.
        param_sets=" ".join(
            f"--param_set {name} {' '.join(str(p) for p in params)}"
            for name, params in METRIC_SETS.items()
        ),
        samples_dir=INIT_SAMPLES_DIR

    output:
        last_sampled_files=expand(INIT_SAMPLES_DIR + "{param_set}/target_test.csv", param_set=INIT_SETS, allow_missing=True),
        split_indices=INIT_SAMPLES_DIR + "split_indices.json"

    conda:
        ENV

    shell:
        """
        python Init.py \
            --input_file_original {input.original_file} \
            --input_file_target {input.target_file} \
            --input_tree_original {params.original_tree} \
            --input_tree_target {params.target_tree} \
            --branches {params.branches} \
            --analysis_params {params.analysis_params} \
            {params.param_sets} \
            --modes {params.modes} \
            --topologies {wildcards.topology} \
            --train_percentage {params.train_percentage} \
            --val_percentage {params.val_percentage} \
            --output_dir {params.samples_dir} \
            --indices_file {output.split_indices}
        """


# The bootstrapped SWD distribution of every set of analysis parameters is obtained the same way,
# whether its samples were written by the initialisation or by the reweighting split: a single
# pair of rules covers them all, with the set named by the {param_set} wildcard.
rule run_bootstrap:
    input:
        target_test=INIT_SAMPLES_DIR + "{param_set}/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "{param_set}/indiv_bootstrap/run_{run_id}.npy"
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"]
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.target_test} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
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


rule custom_dim_analysis:
    input:
        # The samples holding every analysis parameter are already split by 'initialize_analysis':
        # the parameters of interest are simply kept from them, so that the events of a given sample
        # are exactly the ones the initialisation put in it.
        # The last sample written by the initialisation is requested alongside its directory, as the
        # timestamp of a directory does not follow the files it holds: it is not passed to the script.
        init_samples_dir=INIT_SAMPLES_DIR + "all/",
        last_init_sampled_file=INIT_SAMPLES_DIR + "all/target_test.csv"

    params:
        parameters_interest=config["parameters"]["reweighting"],
        tag=config["output"]["tag"]

    output:
        samples_dir=directory(SAMPLES_DIR),
        last_sampled_file=SAMPLES_DIR + "target_test.csv"
    conda:
        ENV
    shell:
        """
        python Splitting_script.py \
            --init_samples_dir {input.init_samples_dir} \
            --parameters_interest {params.parameters_interest} \
            --samples_dir {output.samples_dir}
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
        train_samples_dir=SAMPLES_DIR,
        sample_dirs=expand(INIT_SAMPLES_DIR + "{param_set}/", param_set=PARAM_SETS, allow_missing=True),
        hparam_file=HPS_DIR + "{model}/{model}_hp_{run_id}.json",
        swd_distributions=expand(INIT_SWD_DIR + "{param_set}/swd_distribution_{param_set}.npy",
                                 param_set=PARAM_SETS, allow_missing=True)

    params:
        model="{model}",
        logdir=TENSORBOARD_DIR + "{model}/",
        binning_file=config["parameters"]["binning_file"],
        sample_dir_flags=named_paths_flags("--sample_dir", sample_dirs_of(PARAM_SETS)),
        swd_distribution_flags=named_paths_flags("--swd_distribution", swd_distributions_of(PARAM_SETS)),
        n_directions=config["swd_bootstrapping"]["n_directions"]

    output:
        output_file=TENSORBOARD_DIR + "{model}/run_{run_id}_metrics.csv"

    conda:
        ENV

    shell:
        """
        python Fine_tuning.py \
            --train_sample_dir {input.train_samples_dir} \
            {params.sample_dir_flags} \
            {params.swd_distribution_flags} \
            --model {params.model} \
            --hyperparameters {input.hparam_file} \
            --logdir {params.logdir} \
            --n_directions {params.n_directions} \
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
        samples_dir=SAMPLES_DIR,
        hparam_file=f"set_hyperparameters/{TAG}/{{sample}}/{{topology}}/hyperparameters.json",
        last_sampled_file=SAMPLES_DIR + "target_test.csv"

    params:
        model_list=MODELS,
        model_dir=MODEL_DIR,
        weights_dir=WEIGHTS_DIR
    output:
        save_path_dict=WEIGHTS_DIR + f"weights_path_dict_{REWEIGHTING_SET}.json"

    conda:
        ENV
    shell:
        """
        python Training.py \
            --sample_dir {input.samples_dir} \
            --hparams_dict {input.hparam_file} \
            --save_weights_path {params.weights_dir} \
            --save_model_path {params.model_dir} \
            --model_list {params.model_list} \
            --save_path_dict {output.save_path_dict}
        """


rule compute_metrics:
    input:
        sample_dirs=expand(INIT_SAMPLES_DIR + "{param_set}/", param_set=PARAM_SETS, allow_missing=True),
        last_sampled_files=expand(INIT_SAMPLES_DIR + "{param_set}/target_test.csv",
                                  param_set=PARAM_SETS, allow_missing=True),
        weights_path=WEIGHTS_DIR + f"weights_path_dict_{REWEIGHTING_SET}.json",
        swd_distributions=expand(INIT_SWD_DIR + "{param_set}/swd_distribution_{param_set}.npy",
                                 param_set=PARAM_SETS, allow_missing=True)

    output:
        metrics_file=METRICS_DIR + "metrics.json"

    params:
        binning_file=config["parameters"]["binning_file"],
        sample_dir_flags=named_paths_flags("--sample_dir", sample_dirs_of(PARAM_SETS)),
        swd_distribution_flags=named_paths_flags("--swd_distribution", swd_distributions_of(PARAM_SETS)),
        n_directions=config["swd_bootstrapping"]["n_directions"]

    conda:
        ENV

    shell:
        """
        python Calc_Metrics.py \
            {params.sample_dir_flags} \
            {params.swd_distribution_flags} \
            --weights_paths {input.weights_path} \
            --output_file {output.metrics_file} \
            --compute_chi2 \
            --compute_swd \
            --n_directions {params.n_directions} \
            --binning_file {params.binning_file}
        """


rule make_plots:
    input:
        sample_dir_1D=INIT_SAMPLES_DIR + f"{ALL_PARAMS_SET}/",
        last_sampled_file_1D=INIT_SAMPLES_DIR + f"{ALL_PARAMS_SET}/target_test.csv",
        sample_dir_2D=SAMPLES_DIR,
        last_sampled_file_2D=SAMPLES_DIR + "target_test.csv",
        weights_path=WEIGHTS_DIR + f"weights_path_dict_{REWEIGHTING_SET}.json"

    output:
        histograms_1D=FIG_DIR + "1Dhist.pdf"

    params:
        binning_file=config["parameters"]["binning_file"],
        output_dir=FIG_DIR

    conda:
        ENV

    shell:
        """
        python Make_plots.py \
            --sample_dir_1D {input.sample_dir_1D} \
            --sample_dir_2D {input.sample_dir_2D} \
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
        counts_files=expand(TOPOLOGY_COUNTS_FILE, sample=SAMPLES),
        metrics_files=all_metrics_files,
        figures=all_figure_dirs

# Wildcard-free target running only the topology counting checkpoint on every sample.
# The 'count_topologies' checkpoint itself carries a {sample} wildcard and so cannot be
# asked for on the command line; ask for this rule instead:
#     snakemake count_all_topologies --cores 8
# Running it first makes the counts available to the checkpoints, so that a subsequent
# dry run ('snakemake all -n') can resolve them and list every job of the workflow.
rule count_all_topologies:
    input:
        counts_files=expand(TOPOLOGY_COUNTS_FILE, sample=SAMPLES)

