This repository allows one to perform a reweighting analysis on any given Nuisance file. 

[Initializing if you don't have conda installed]
If you DO have conda installed, simply add --use-conda at the end of any command including snakemake.
If you DO NOT have conda installed, you should first create a virtual environment with the following command :
    python3 -m venv .venv
    source .venv/bin/activate
and then install the necessary librairies with :
    pip install -r requirements.txt
Please use the snakemake commands WITHOUT --use-conda at the end

After cloning the repository, one should first download the input files. For example, the Neut and Genie files used in previous analysis were taken from:
    -https://portal.nersc.gov/project/nuisance/MC_IOP_review/GENIEv3_G18_10b_00_000/, especially : T2KND_FHC_numu_H2O_GENIEv3_G18_10b_00_000_1M_0000_NUISFLAT.root
    -https://portal.nersc.gov/project/nuisance/MC_IOP_review/NEUT562/, especially : T2KND_FHC_numu_H2O_NEUT562_1M_0000_NUISFLAT.root

After downloading the needed files, one should place them in the appropriate directory (ex : GENIE_file and NEUT_file directory, but feel free to create your own directory). 
Then, open the file named 'config.yaml', and change the 'neut_file' and 'genie_file' path to match the relative path of the files you want to reweight. Please note that the original distribution will be the one given to 'neut_file' and the target will be the one given to 'genie_file'.
Also in the 'config.yaml' file, you can change several parameters for your analysis : 

In the 'analysis' section, you can specify 
    -the interaction mode number (default is 1 for CCQE), 
    -the topologies to analyse, given by name and picked from : CC0pi, CC1pipm, CC1pi0, CCNpi, CCgamma, CCOther, NCInc and rest.
     A separate training is run for each listed topology, and the results are saved in a sub-directory named after the topology.
     Before anything else, the events of each topology are counted in every sample (and saved in
     'saved_samples/{tag}/{sample}/topology_counts.json'). If a topology holds too few events in either the original or the
     target file of a sample to be split into non-empty training, validation and test samples, a warning is printed and no
     training is carried out for that topology in that sample. The minimum number of events follows from the train and
     validation percentages set below (3 events with the default 0.4/0.4 split, since int(0.4*2) = 0 would leave the
     training and validation samples empty). The same topology is still trained on in the samples where it is filled enough.
    -the PDG number of the interacting neutrino (default is 14 for muon neutrino)
    -the train_percentage which defines the amount of data that go into your training sample (default is 0.4, should be between 0 and 1)
    -the val_percentage which defines the amount of data that go into your validation sample (default is 0.4, should be between 0 and 1)
    Please make sure that the sum of the percentages does not exceed 1. (The amount of datat going into the test sample is automatically computed)

In the 'parameters'/'reweighting' section, you can specifiy the parameters that you want to use to train the different methods on.
Please pick from the parameters listed in the 'parameters'/'all' section.

The metrics are computed for several named sets of parameters, listed in the 'parameters'/'metric_sets' section
(for example '3D' and '8D'). Every parameter of a set must also be listed in the 'parameters'/'all' section, which
holds every analysis parameter. Adding or removing a set there is all it takes: the samples, the bootstrapped SWD
distribution and the metrics of the new set are produced by the same rules as the other ones, and its metrics are
written under keys named after it (ex : 'swd_3D', 'p_value_3D', 'chi2_3D'). The 'all' set and the set of the
reweighting parameters (named 'custom_{Dim}D') are always added to the configured ones.

In the 'models'/'model_list' section, you can specifiy the models that you want to use.
Please pick from the following list : ['binning', 'XGB'].

In the 'models'/'grid_file' section, you can give the json file holding the hyperparameter grid every model is
fine-tuned over (default is 'hyperparameter_grids.json'). It holds, for each model, the list of values every
hyperparameter is scanned over, for example :
{"binning": {"n_bins": [5, 10], "n_neighs": [0, 1]},
 "XGB": {"n_estimators": [80, 100], "max_depth": [3], "learning_rate": [0.075], "subsample": [1], "early_stopping_rounds": [10]}}
The grid of a model is the cartesian product of those lists (four sets of hyperparameters for the 'binning' example
above), so the number of fine-tuning runs of a model follows from the grid file and never has to be given.

In the 'models'/'selection_metric' section, you can give, for each model, the metric its best set of hyperparameters
is picked with and whether that metric is minimised ('min') or maximised ('max'). The metrics are named after the
parameter sets above (ex : 'SWD_3D', 'p_value_8D', 'chi2_dof_all').

In the 'dimensions'/'dim' section, you can specify the dimension of the parameters of interest. This is purely for folder creation, so that each combination of parameters can be created.
Default is a number, but you may give any value as argument.

In the 'output'/'tag' you can specify the tag that you want to give to your analysis. The files will be created in the different folders in subfolder with the given tag.
For example this can include parts of the name of the reweighted distributions. 
Please use strings.

[Hyperparameters]

The hyperparameters of the models are fine-tuned by the workflow itself: one run of 'Fine_tuning.py' is carried out
per point of the grid given in the 'models'/'grid_file' file, and 'Gather_fine_tuning.py' then keeps, for each model,
the set of hyperparameters giving the best value of the metric given in the 'models'/'selection_metric' section. The
chosen sets are written in 'set_hyperparameters/{tag}/{sample}/{topology}/hyperparameters.json' and are the ones the
final training uses. Nothing has to be written by hand: to change the hyperparameters that are scanned, change the
grid file.

Once all the above steps are completed, you are ready to run the analysis.
To run a complete analysis, including samples creation, model training, metrics evaluation and plotting, enter the following command in your terminal: 

snakemake all --cores 8  (--use-conda  (only if you use conda))

This runs every sample found in the input directories, for every topology listed in the config file.
To run a single sample/topology combination, you can also ask for one output file directly, for example:

snakemake saved_metrics/{tag}/{sample}/{topology}/custom_{Dim}D/metrics.json --cores 8  (--use-conda  (only if you use conda))

where {tag} is the tag given in the 'output'/'tag' section, {sample} is the relative path of the sample (ex : FHC/numu/H2O),
 {topology} is one of the topologies listed in the 'analysis'/'topologies' section (ex : CC0pi), and {Dim} is the number
of parameters listed in the 'parameters'/'reweighting' section of the 'config.yaml' file.
The metrics are written by 'Calc_Metrics.py' in 'saved_metrics', and the plots by 'Make_plots.py' in 'saved_figures':
asking for one of them does not run the other.
Once finished, you can explore the different 'saved' folders containing the samples, models, metrics and plots.

Note on dry runs ('snakemake all -n'): the topology counting is a snakemake checkpoint, which means the jobs
that depend on it (all the trainings and metrics) can only be listed once the counts exist. A dry run creates no
file, so it stops at the counting jobs. To see the whole list of jobs, first run the (cheap) counting on every
sample, then ask for the dry run:

snakemake count_all_topologies --cores 8  (--use-conda  (only if you use conda))
snakemake all -n

The counting is not repeated afterwards, as its output files are then up to date.

Congrats, you ran your first analysis !

[Weights]

Every event carries a pre-weight, the product of the 'RWWeight' and 'fScaleFactor' branches of the input files.
It is stored as a trailing 'PreWeight' column of every sample csv file, and the original and target distributions
are pre-weighted by it before the models are trained. The models therefore learn the reweighting taking the
pre-weighted original distribution to the pre-weighted target distribution.

The weights saved in 'saved_weights' are the multipliers of that reweighting: they do not include the pre-weight
of the events. To get the absolute weight of an event, multiply the saved weight by the 'PreWeight' column of the
sample it belongs to. The multipliers carry the normalisation, so that the sum of the absolute weights of the
reweighted original sample matches the sum of the pre-weights of the target sample.

[Samples]

The samples are created once per sample and topology by 'Init.py', which splits the events into a training, a
validation and a test sample and writes them for every analysis parameter (the 'all' set) and for every set listed
in the 'parameters'/'metric_sets' section of the config file, each in a sub-directory named after the set. The
indices of the split are saved next to them, in 'split_indices.json'. The samples holding the parameters the models
are trained on ('parameters'/'reweighting') are then obtained by 'Splitting_script.py', which simply keeps the
corresponding columns of the 'all' samples: the input files are only ever read once, and the split is never
recomputed.

[Input files]

'Clean_input_tree.py' strips the input ROOT files of everything but the directories that are needed, for both the
GENIE files (which keep the '2024NoGSF' directory) and the NEUT files (which keep 'BANFF_PRE' and require
'BANFF_POST' to be present as well), for example :

python Clean_input_tree.py GENIE_files GENIE_clean --keep-dir 2024NoGSF
python Clean_input_tree.py NEUT_files NEUT_clean --keep-dir BANFF_PRE --require-dir BANFF_POST





