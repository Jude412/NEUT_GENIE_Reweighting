configfile: "config.yaml"

import glob
import json
import os

ENV = "environment.yaml"
TAG = config["output"]["tag"]
DIM = len(config["parameters"]["reweighting"])
NUMBER_OF_SETS = config["models"]["number_model_hyperparameters_sets"]
MODELS = config["models"]["model_list"]
MODES = config["analysis"]["modes"]
TOPOLOGIES = config["analysis"]["topologies"]
RUNS = range(config["swd_bootstrapping"]["runs"])

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

INIT_SAMPLES_DIR = f"saved_samples/{TAG}/{{sample}}/"
INIT_SWD_DIR     = f"saved_swd_distribution/{TAG}/{{sample}}/"
SAMPLES_DIR      = f"saved_samples/{TAG}/{{sample}}/custom_{DIM}D/"
SWD_DIR          = f"saved_swd_distribution/{TAG}/{{sample}}/custom_{DIM}D/"
MODEL_DIR        = f"saved_models/{TAG}/{{sample}}/custom_{DIM}D/"
WEIGHTS_DIR      = f"saved_weights/{TAG}/{{sample}}/custom_{DIM}D/"
FIG_DIR          = f"saved_figures/{TAG}/{{sample}}/custom_{DIM}D/"
HPS_DIR          = f"hps/{TAG}/"
TENSORBOARD_DIR  = f"TensorBoard/{TAG}/{{sample}}/custom_{DIM}D/"

# Hyperparameter grid files are sample-independent (shared search space).
HPS_FILES = []
for model in MODELS:
    for run_id in range(NUMBER_OF_SETS[model]):
        HPS_FILES.append(os.path.join(HPS_DIR, f"{model}/{model}_hp_{run_id}.json"))

# Wildcard constraint: sample paths contain only word characters and slashes.
wildcard_constraints:
    sample="[^.]+"


rule initialize_analysis:
    input:
        original_file=original_file_for,
        target_file=target_file_for

    params:
        modes=MODES,
        topologies=TOPOLOGIES,
        original_tree=config["inputs"]["original_tree"],
        target_tree=config["inputs"]["target_tree"],
        # neutrino_PDG=config["analysis"]["neutrino_PDG"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"],
        branches=config["inputs"]["branches"],
        analysis_params=config["parameters"]["all"],
        params_8D=config["parameters"]["8D"],
        params_3D=config["parameters"]["3D"]

    output:
        samples_dir_3D=directory(INIT_SAMPLES_DIR + "3D/"),
        samples_dir_8D=directory(INIT_SAMPLES_DIR + "8D/"),
        samples_dir_all=directory(INIT_SAMPLES_DIR + "all/"),
        original_test_8D=INIT_SAMPLES_DIR + "8D/original_test.csv",
        target_test_8D=INIT_SAMPLES_DIR + "8D/target_test.csv",
        original_test_all=INIT_SAMPLES_DIR + "all/original_test.csv",
        target_test_all=INIT_SAMPLES_DIR + "all/target_test.csv",
        last_sampled_file_3D=INIT_SAMPLES_DIR + "3D/target_test.csv"

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
            --params_8D {params.params_8D} \
            --params_3D {params.params_3D} \
            --modes {params.modes} \
            --topologies {params.topologies} \
            --train_percentage {params.train_percentage} \
            --val_percentage {params.val_percentage} \
            --output_dir_samples_3D {output.samples_dir_3D} \
            --output_dir_samples_8D {output.samples_dir_8D} \
            --output_dir_samples_all {output.samples_dir_all}
        """


rule run_initial_bootstrap_3D:
    input:
        samples_dir=INIT_SAMPLES_DIR + "3D/target_test.csv",
        last_sampled_file=INIT_SAMPLES_DIR + "3D/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "3D/indiv_bootstrap/run_{run_id}.npy"
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"]
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {params.n_directions} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_bootstrap_3D:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/3D/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=INIT_SWD_DIR + "3D/swd_distribution_3D.npy"
    conda:
        ENV
    shell:
        """
        python aggregate.py \
            --input {input} \
            --output {output.final_file}
        """


rule run_initial_bootstrap_8D:
    input:
        samples_dir=INIT_SAMPLES_DIR + "8D/target_test.csv",
        last_sampled_file=INIT_SAMPLES_DIR + "8D/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "8D/indiv_bootstrap/run_{run_id}.npy"
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"]
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {params.n_directions} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_bootstrap_8D:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/8D/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=INIT_SWD_DIR + "8D/swd_distribution_8D.npy"
    conda:
        ENV
    shell:
        """
        python aggregate.py \
            --input {input} \
            --output {output.final_file}
        """

rule run_initial_bootstrap_all:
    input:
        samples_dir=INIT_SAMPLES_DIR + "all/target_test.csv",
        last_sampled_file=INIT_SAMPLES_DIR + "all/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "all/indiv_bootstrap/run_{run_id}.npy"
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"]
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {params.n_directions} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_bootstrap_all:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/all/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=INIT_SWD_DIR + "all/swd_distribution_all.npy"
    conda:
        ENV
    shell:
        """
        python aggregate.py \
            --input {input} \
            --output {output.final_file}
        """

rule custom_dim_analysis:
    input:
        original_file=original_file_for,
        target_file=target_file_for

    params:
        modes=MODES,
        topologies=TOPOLOGIES,
        original_tree=config["inputs"]["original_tree"],
        target_tree=config["inputs"]["target_tree"],
        # neutrino_PDG=config["analysis"]["neutrino_PDG"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"],
        branches=config["inputs"]["branches"],
        analysis_params=config["parameters"]["all"],
        parameters_interest=config["parameters"]["reweighting"],
        tag=config["output"]["tag"]

    output:
        samples_dir=directory(SAMPLES_DIR),
        last_sampled_file=SAMPLES_DIR + "target_test.csv"
    conda:
        ENV
    shell:
        """
        python Splitting-script.py \
            --input_file_original {input.original_file} \
            --input_file_target {input.target_file} \
            --input_tree_original {params.original_tree} \
            --input_tree_target {params.target_tree} \
            --branches {params.branches} \
            --analysis_params {params.analysis_params} \
            --modes {params.modes} \
            --topologies {params.topologies} \
            --train_percentage {params.train_percentage} \
            --val_percentage {params.val_percentage} \
            --parameters_interest {params.parameters_interest} \
            --samples_dir {output.samples_dir}
        """

rule run_custom_bootstrap:
    input:
        samples_dir=SAMPLES_DIR + "target_test.csv",
    output:
        output_file=SWD_DIR + "indiv_bootstrap/run_{run_id}.npy"
    params:
        n_directions=config["swd_bootstrapping"]["n_directions"]
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {params.n_directions} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_custom_bootstrap:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/custom_{DIM}D/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=SWD_DIR + f"swd_distribution_custom_{DIM}D.npy"
    conda:
        ENV
    shell:
        """
        python aggregate.py \
            --input {input} \
            --output {output.final_file}
        """

rule prepare_hps:
    output:
        hps_files=HPS_FILES
    conda:
        ENV
    shell:
        """
        python list_parambinning.py --output_dir {HPS_DIR}
        python list_paramnn.py  --output_dir {HPS_DIR}
        python list_paramgbr.py  --output_dir {HPS_DIR}
        python list_paramxgb.py  --output_dir {HPS_DIR}
        """

rule single_fine_tuning:
    input:
        train_samples_dir=SAMPLES_DIR,
        init_samples_dir_3D=INIT_SAMPLES_DIR + "3D/",
        init_samples_dir_8D=INIT_SAMPLES_DIR + "8D/",
        init_samples_dir_all=INIT_SAMPLES_DIR + "all/",
        hparam_file=HPS_DIR + "{model}/{model}_hp_{run_id}.json",
        swd_distribution_3D=INIT_SWD_DIR + "3D/swd_distribution_3D.npy",
        swd_distribution_8D=INIT_SWD_DIR + "8D/swd_distribution_8D.npy",
        swd_distribution_all=INIT_SWD_DIR + "all/swd_distribution_all.npy",
        custom_swd_distribution=SWD_DIR + f"swd_distribution_custom_{DIM}D.npy"

    params:
        model="{model}",
        logdir=TENSORBOARD_DIR + "{model}/",
        binning_file=config["parameters"]["binning_file"],
        analysis_params=config["parameters"]["all"],
        params_8D=config["parameters"]["8D"],
        params_3D=config["parameters"]["3D"],
        interest_params=config["parameters"]["reweighting"],
        n_directions=config["swd_bootstrapping"]["n_directions"]

    output:
        output_file=TENSORBOARD_DIR + "{model}/run_{run_id}_metrics.csv"

    conda:
        ENV

    shell:
        """
        python Fine-tuning.py \
            --train_sample_dir {input.train_samples_dir} \
            --sample_dir_3D {input.init_samples_dir_3D} \
            --sample_dir_8D {input.init_samples_dir_8D} \
            --sample_dir_all {input.init_samples_dir_all} \
            --swd_distribution_3D {input.swd_distribution_3D} \
            --swd_distribution_8D {input.swd_distribution_8D} \
            --swd_distribution_all {input.swd_distribution_all} \
            --model {params.model} \
            --hyperparameters {input.hparam_file} \
            --logdir {params.logdir} \
            --custom_swd_distribution {input.custom_swd_distribution} \
            --analysis_params {params.analysis_params} \
            --params_8D {params.params_8D} \
            --params_3D {params.params_3D} \
            --params_interest {params.interest_params} \
            --n_directions {params.n_directions} \
            --binning_file {params.binning_file} \
            --output_file {output.output_file}
        """

rule fine_tuning:
    input:
        lambda wc: [
            f"TensorBoard/{TAG}/{wc.sample}/custom_{DIM}D/{model}/run_{run_id}_metrics.csv"
            for model in MODELS
            for run_id in range(NUMBER_OF_SETS[model])
        ]
    output:
        output_file=f"set_hyperparameters/{TAG}/{{sample}}/hyperparameters.json"
    params:
        logdir=TENSORBOARD_DIR,
        model_list=MODELS
    conda:
        ENV
    shell:
        """
        python gather_fine_tuning.py \
            --input_dir {params.logdir} \
            --output_file {output.output_file} \
            --model_list {params.model_list}
        """


rule train_models:
    input:
        samples_dir=SAMPLES_DIR,
        hparam_file=f"set_hyperparameters/{TAG}/{{sample}}/hyperparameters.json",
        last_sampled_file=SAMPLES_DIR + "target_test.csv"

    params:
        model_list=config["models"]["model_list"],
        parameters_interest=config["parameters"]["reweighting"],
        model_list_str=config["models"]["model_list"]
    output:
        model_dir=directory(MODEL_DIR),
        weights_dir=directory(WEIGHTS_DIR),
        save_weight_path_dict=WEIGHTS_DIR + f"weights_path_dict_custom_{DIM}D.json"

    conda:
        ENV
    shell:
        """
        python Training.py \
            --original_train {input.samples_dir}/original_train.csv \
            --original_val {input.samples_dir}/original_val.csv \
            --original_test {input.samples_dir}/original_test.csv \
            --target_train {input.samples_dir}/target_train.csv \
            --target_val {input.samples_dir}/target_val.csv \
            --target_test {input.samples_dir}/target_test.csv \
            --hparams_dict {input.hparam_file} \
            --save_weights_path {output.weights_dir} \
            --save_model_path {output.model_dir} \
            --model_list {params.model_list_str} \
            --save_weight_path_dict {output.save_weight_path_dict} \
        """


rule compute_metrics_plots:
    input:
        original_test=INIT_SAMPLES_DIR + "all/original_test.csv",
        target_test=INIT_SAMPLES_DIR + "all/target_test.csv",
        weights_path=WEIGHTS_DIR + f"weights_path_dict_custom_{DIM}D.json",
        swd_dist_file_3D=INIT_SWD_DIR + "3D/swd_distribution_3D.npy",
        swd_dist_file_8D=INIT_SWD_DIR + "8D/swd_distribution_8D.npy",
        swd_dist_file_all=INIT_SWD_DIR + "all/swd_distribution_all.npy",
        swd_dist_file=SWD_DIR + f"swd_distribution_custom_{DIM}D.npy"

    output:
        output_dir=directory(FIG_DIR),
        metrics_file=FIG_DIR + "metrics.json"

    params:
        parameters_interest=config["parameters"]["reweighting"],
        binning_file=config["parameters"]["binning_file"],
        analysis_params=config["parameters"]["all"],
        params_8D=config["parameters"]["8D"],
        params_3D=config["parameters"]["3D"],
        n_directions=config["swd_bootstrapping"]["n_directions"]

    conda:
        ENV

    shell:
        """
        python Calc_Metrics.py \
            --original_test {input.original_test} \
            --target_test {input.target_test} \
            --weights_paths {input.weights_path} \
            --output_file {output.output_dir} \
            --make_1D_plots \
            --no-make_2D_plots \
            --compute_chi2 \
            --compute_swd \
            --analysis_params {params.analysis_params} \
            --params_8D {params.params_8D} \
            --params_3D {params.params_3D} \
            --interest_params {params.parameters_interest} \
            --custom_swd_distribution {input.swd_dist_file} \
            --swd_distribution_3D {input.swd_dist_file_3D} \
            --swd_distribution_8D {input.swd_dist_file_8D} \
            --swd_distribution_all {input.swd_dist_file_all} \
            --n_directions {params.n_directions} \
            --binning_file {params.binning_file}
        """

rule all:
    input:
        expand(
            f"saved_figures/{TAG}/{{sample}}/custom_{DIM}D/metrics.json",
            sample=SAMPLES
        )
    
