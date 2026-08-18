configfile: "config.yaml"

import glob
import json
import os

ENV = "environment.yaml"
TAG = config["output"]["tag"]
DIM = config["dimensions"]["dim"]
NUMBER_OF_SETS = config["models"]["number_model_hyperparameters_sets"]
MODELS = config["models"]["model_list"]
MODES = config["analysis"]["modes"]
MODES_v2 = config["analysis"]["modes_v2"]
RUNS = range(config["swd_bootstrapping"]["runs"])
NDIR = config["swd_bootstrapping"]["n_directions"]

NEUT_DIR = config["paths"]["neut_dir"]
GENIE_DIR = config["paths"]["genie_dir"]

# ---------------------------------------------------------------------------
# Sample discovery
# Scan NEUT_DIR recursively for .root files; keep any whose leaf directory
# also contains a .root file under the same relative path in GENIE_DIR.
# SAMPLES is a list of relative directory paths, e.g. ["FHC/numu/H2O"].
# ---------------------------------------------------------------------------

def get_samples():
    neut_files = glob.glob(os.path.join(NEUT_DIR, "**/*.root"), recursive=True)
    seen = set()
    samples = []
    for nf in neut_files:
        rel = os.path.relpath(os.path.dirname(nf), NEUT_DIR)
        if rel not in seen and glob.glob(os.path.join(GENIE_DIR, rel, "*.root")):
            seen.add(rel)
            samples.append(rel)
    return samples

SAMPLES = get_samples()

# ---------------------------------------------------------------------------
# Input functions – resolve the single .root file inside a leaf directory.
# ---------------------------------------------------------------------------

def neut_file_for(wildcards):
    files = glob.glob(os.path.join(NEUT_DIR, wildcards.sample, "*.root"))
    return files[0]

def genie_file_for(wildcards):
    files = glob.glob(os.path.join(GENIE_DIR, wildcards.sample, "*.root"))
    return files[0]

# ---------------------------------------------------------------------------
# Per-sample path helpers  (all output paths are namespaced by {sample})
# ---------------------------------------------------------------------------

INIT_SAMPLES_DIR = f"saved_samples/{TAG}/{{sample}}/"
INIT_SWD_DIR     = f"saved_swd_distribution/{TAG}/{{sample}}/"
SAMPLES_DIR      = f"saved_samples/{TAG}/{{sample}}/{DIM}D/"
SWD_DIR          = f"saved_swd_distribution/{TAG}/{{sample}}/{DIM}D/"
MODEL_DIR        = f"saved_models/{TAG}/{{sample}}/{DIM}D/"
WEIGHTS_DIR      = f"saved_weights/{TAG}/{{sample}}/{DIM}D/"
FIG_DIR          = f"saved_figures/{TAG}/{{sample}}/{DIM}D/"
HPS_DIR          = f"hps/{TAG}/"
TENSORBOARD_DIR  = f"TensorBoard/{TAG}/{{sample}}/{DIM}D/"

# Hyperparameter grid files are sample-independent (shared search space).HPS_FILES = []
for model in MODELS:
    for run_id in range(NUMBER_OF_SETS[model]):
        HPS_FILES.append(os.path.join(HPS_DIR, f"{model}/{model}_hp_{run_id}.json"))

# Wildcard constraint: sample paths contain only word characters and slashes.
wildcard_constraints:
    sample="[^.]+"


rule initialize_analysis:
    input:
        NEUTfile=neut_file_for,
        GENIEfile=genie_file_for

    params:
        modes=MODES,
        modes_v2=MODES_v2,
        # neutrino_PDG=config["analysis"]["neutrino_PDG"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"]

    output:
        samples_dir_3D=directory(INIT_SAMPLES_DIR + "3D/"),
        samples_dir_8D=directory(INIT_SAMPLES_DIR + "8D/"),
        samples_dir_21D=directory(INIT_SAMPLES_DIR + "21D/"),
        original_test_8D=INIT_SAMPLES_DIR + "8D/original_test.csv",
        target_test_8D=INIT_SAMPLES_DIR + "8D/target_test.csv",
        original_test_21D=INIT_SAMPLES_DIR + "21D/original_test.csv",
        target_test_21D=INIT_SAMPLES_DIR + "21D/target_test.csv",
        last_sampled_file_3D=INIT_SAMPLES_DIR + "3D/target_test.csv"

    conda:
        ENV

    shell:
        """
        python Init.py \
            --input_file_NEUT {input.NEUTfile} \
            --input_file_GENIE {input.GENIEfile} \
            --modes {params.modes} \
            --modes_v2 {params.modes_v2} \
            --train_percentage {params.train_percentage} \
            --val_percentage {params.val_percentage} \
            --output_dir_samples_3D {output.samples_dir_3D} \
            --output_dir_samples_8D {output.samples_dir_8D} \
            --output_dir_samples_21D {output.samples_dir_21D}
        """


rule run_initial_bootstrap_3D:
    input:
        samples_dir=INIT_SAMPLES_DIR + "3D/target_test.csv",
        last_sampled_file=INIT_SAMPLES_DIR + "3D/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "3D/indiv_bootstrap/run_{run_id}.npy"
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {NDIR} \
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
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {NDIR} \
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

rule run_initial_bootstrap_21D:
    input:
        samples_dir=INIT_SAMPLES_DIR + "21D/target_test.csv",
        last_sampled_file=INIT_SAMPLES_DIR + "21D/target_test.csv"
    output:
        output_file=INIT_SWD_DIR + "21D/indiv_bootstrap/run_{run_id}.npy"
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {NDIR} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_bootstrap_21D:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/21D/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=INIT_SWD_DIR + "21D/swd_distribution_21D.npy"
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
        NEUTfile=neut_file_for,
        GENIEfile=genie_file_for

    params:
        modes=MODES,
        modes_v2=MODES_v2,
        # neutrino_PDG=config["analysis"]["neutrino_PDG"],
        train_percentage=config["analysis"]["train_percentage"],
        val_percentage=config["analysis"]["val_percentage"],
        parameters_interest=config["features"]["parameters_interest"],
        tag=config["output"]["tag"]

    output:
        samples_dir=directory(SAMPLES_DIR),
        last_sampled_file=SAMPLES_DIR + "target_test.csv"
    conda:
        ENV
    shell:
        """
        python Splitting-script.py \
            --input_file_NEUT {input.NEUTfile} \
            --input_file_GENIE {input.GENIEfile} \
            --modes {params.modes} \
            --modes_v2 {params.modes_v2} \
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
    conda:
        ENV
    shell:
        """
        python Bootstrap_swd.py \
            --distribution {input.samples_dir} \
            --output_dir $(dirname {output.output_file})/ \
            --output_file {output.output_file} \
            --n_directions {NDIR} \
            --random_seed {wildcards.run_id}
        """

rule aggregate_custom_bootstrap:
    input:
        lambda wc: expand(
            f"saved_swd_distribution/{TAG}/{wc.sample}/{DIM}D/indiv_bootstrap/run_{{run_id}}.npy",
            run_id=RUNS
        )
    output:
        final_file=SWD_DIR + f"swd_distribution_{DIM}D.npy"
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
        init_samples_dir_21D=INIT_SAMPLES_DIR + "21D/",
        hparam_file=HPS_DIR + "{model}/{model}_hp_{run_id}.json",
        swd_distribution_3D=INIT_SWD_DIR + "3D/swd_distribution_3D.npy",
        swd_distribution_8D=INIT_SWD_DIR + "8D/swd_distribution_8D.npy",
        swd_distribution_21D=INIT_SWD_DIR + "21D/swd_distribution_21D.npy",
        custom_swd_distribution=SWD_DIR + f"swd_distribution_{DIM}D.npy"

    params:
        model="{model}",
        logdir=TENSORBOARD_DIR + "{model}/",
        binning_file=config["features"]["binning_file"],
        interest_params=config["features"]["parameters_interest"]

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
            --sample_dir_21D {input.init_samples_dir_21D} \
            --swd_distribution_3D {input.swd_distribution_3D} \
            --swd_distribution_8D {input.swd_distribution_8D} \
            --swd_distribution_21D {input.swd_distribution_21D} \
            --model {params.model} \
            --hyperparameters {input.hparam_file} \
            --logdir {params.logdir} \
            --custom_swd_distribution {input.custom_swd_distribution} \
            --params_interest {params.interest_params} \
            --n_directions {NDIR} \
            --binning_file {params.binning_file} \
            --output_file {output.output_file}
        """

rule fine_tuning:
    input:
        lambda wc: [
            f"TensorBoard/{TAG}/{wc.sample}/{DIM}D/{model}/run_{run_id}_metrics.csv"
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
        parameters_interest=config["features"]["parameters_interest"],
        model_list_str=config["models"]["model_list"]
    output:
        model_dir=directory(MODEL_DIR),
        weights_dir=directory(WEIGHTS_DIR),
        save_weight_path_dict=WEIGHTS_DIR + f"weights_path_dict_{DIM}D.json"

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
        original_test=INIT_SAMPLES_DIR + "21D/original_test.csv",
        target_test=INIT_SAMPLES_DIR + "21D/target_test.csv",
        weights_path=WEIGHTS_DIR + f"weights_path_dict_{DIM}D.json",
        swd_dist_file_3D=INIT_SWD_DIR + "3D/swd_distribution_3D.npy",
        swd_dist_file_8D=INIT_SWD_DIR + "8D/swd_distribution_8D.npy",
        swd_dist_file_21D=INIT_SWD_DIR + "21D/swd_distribution_21D.npy",
        swd_dist_file=SWD_DIR + f"swd_distribution_{DIM}D.npy"

    output:
        output_dir=directory(FIG_DIR),
        metrics_file=FIG_DIR + "metrics.json"

    params:
        parameters_interest=config["features"]["parameters_interest"],
        binning_file=config["features"]["binning_file"]

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
            --interest_params {params.parameters_interest} \
            --custom_swd_distribution {input.swd_dist_file} \
            --swd_distribution_3D {input.swd_dist_file_3D} \
            --swd_distribution_8D {input.swd_dist_file_8D} \
            --swd_distribution_21D {input.swd_dist_file_21D} \
            --n_directions {NDIR} \
            --binning_file {params.binning_file}
        """

rule all:
    input:
        expand(
            f"saved_figures/{TAG}/{{sample}}/{DIM}D/metrics.json",
            sample=SAMPLES
        )
    
