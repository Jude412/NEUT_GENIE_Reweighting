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

In the 'features'/'parameters_interest' section, you can specifiy the parameters that you want to use to train the different methods on.
Please pick from the following list : ["Enu_true", "ELep", "CosLep", "Q2", "q0", "q3", "W", "Eav"].

In the 'models'/'model_list' section, you can specifiy the models that you want to use.
Please pick from the following list : ['binning', 'NN', 'GBR', 'XGB'].

In the 'dimensions'/'dim' section, you can specify the dimension of the parameters of interest. This is purely for folder creation, so that each combination of parameters can be created.
Default is a number, but you may give any value as argument.

In the 'output'/'tag' you can specify the tag that you want to give to your analysis. The files will be created in the different folders in subfolder with the given tag.
For example this can include parts of the name of the reweighted distributions. 
Please use strings.

[The following steps are hacky and will be removed when the grid search feature will be ready]
///
In the 'set_hyperparameters' directory, create a sub-directory with the same tag that you gave in the 'config.yaml' file.
Inside this directory, create a file named 'hyperparameters.json'. This file should contain a dictionnary, containing a dictionnary of the set of hyperparameters you want the different method to be built with. 
For example : 
{"binning": {"n_bins" : 10, "n_neighs" : 0},
"NN" : {"n_layers" : 2, "n_neurons" : 15, "epochs" : 100, "batch_size" : 4096, "activation_function" : "tanh"},
"GBR" : {"n_estimators": 110, "learning_rate": 0.03, "max_depth": 4, "min_samples_leaf": 90, "loss_regularization":3},
"XGB" : {"n_estimators" : 80, "gamma" : 0, "max_depth" : 3, "learning_rate" : 0.075, "subsample" : 1, "early_stopping_rounds" : 10}}

Please give as many sets as different method you want to train.
///

Once all the above steps are completed, you are ready to run the analysis.
To run a complete analysis, including samples creation, model training, metrics evaluation and plotting, enter the following command in your terminal: 

snakemake all --cores 8  (--use-conda  (only if you use conda))

This runs every sample found in the input directories, for every topology listed in the config file.
To run a single sample/topology combination, you can also ask for one output file directly, for example:

snakemake saved_figures/{tag}/{sample}/{topology}/custom_{Dim}D/metrics.json --cores 8  (--use-conda  (only if you use conda))

where {tag} is the tag given in the 'output'/'tag' section, {sample} is the relative path of the sample (ex : FHC/numu/H2O),
{topology} is one of the topologies listed in the 'analysis'/'topologies' section (ex : CC0pi) and {Dim} is the number of
dimensions given in the 'dimensions'/'dim' section of the 'config.yaml' file.
Once finished, you can explore the different 'saved' folders containing the samples, models, metrics and plots.

Note on dry runs ('snakemake all -n'): the topology counting is a snakemake checkpoint, which means the jobs
that depend on it (all the trainings and metrics) can only be listed once the counts exist. A dry run creates no
file, so it stops at the counting jobs. To see the whole list of jobs, first run the (cheap) counting on every
sample, then ask for the dry run:

snakemake count_all_topologies --cores 8  (--use-conda  (only if you use conda))
snakemake all -n

The counting is not repeated afterwards, as its output files are then up to date.

Congrats, you ran your first analysis !





