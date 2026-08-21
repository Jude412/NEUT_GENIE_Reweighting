"""The goal of this script is to train any  model to reweight the 'original' distribution into the 'target' distribution.
The hyperparameters and metrics are saved in a tensorboard log file as well as a csv format."""

#imports
import json

from Metrics_ndim import  compute_swd, compute_p_value, chi2_dof, chi2_hist_axis
from Train_predict import train_binning, predict_binning, train_NN, predict_NN, train_GBR, predict_GBR, train_XGB, predict_XGB
import numpy as np
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train an XGBoost model to reweight the 'original' distribution into the 'target' distribution.")
    args.add_argument('--train_sample_dir', type=str, required=False, help='Directory where the training original and target samples are stored in csv format.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/7D/")
    args.add_argument('--sample_dir_3D', type=str, required=False, help='Directory where the 3D original and target samples are stored in csv format.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/3D/")
    args.add_argument('--sample_dir_8D', type=str, required=False, help='Directory where the 8D original and target samples are stored in csv format.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/8D/")
    args.add_argument('--sample_dir_all', '--sample_dir_21D', dest='sample_dir_all', type=str, required=False, help='Directory where the original and target samples containing all configured analysis parameters are stored in csv format.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/all/")
    args.add_argument('--model', type=str, required=True, choices=['binning', 'NN', 'GBR', 'XGB'], help='The reweighting model to train.')
    args.add_argument('--hyperparameters', type=str, required=True, help="Path to JSON file containing hyperparameters to train the XGBoost model.")
    args.add_argument('--logdir', type=str, required=False, default="/vols/dune/jmm224/t2knova/reweighting/TensorBoard/test_run2", help="The directory where the tensorboard log file will be saved.")
    args.add_argument("--custom_swd_distribution", type = str, help = "The distribution to compute the Training-dim SWD p-value for.")
    args.add_argument("--swd_distribution_3D", type = str, help = "The distribution to compute the 3D SWD p-value for.")
    args.add_argument("--swd_distribution_8D", type = str, help = "The distribution to compute the 8D SWD p-value for.")
    args.add_argument("--swd_distribution_all", "--swd_distribution_21D", dest="swd_distribution_all", type = str, help = "The distribution to compute the all-parameters SWD p-value for.")
    args.add_argument("--analysis_params", nargs="+", help="List of parameters to extract from the branches for the analysis.")
    args.add_argument("--params_8D", nargs="+", help="List of parameters to use in 8D samples.")
    args.add_argument("--params_3D", nargs="+", help="List of parameters to use in 3D samples.")
    args.add_argument("--params_interest", nargs='+', help="List of parameter names used in the training.")
    args.add_argument('--n_directions', type=int, default=500, help="Number of directions to draw for each bootstrap")
    args.add_argument("--binning_file", type=str, help="Path to the json file containing the binning information for each parameter.")
    args.add_argument("--output_file", type = str, help = "The path to the csv file where the hyperparameters and metrics will be saved in addition to the tensorboard log file.")
    args = args.parse_args()

    Index_params_interest = [args.analysis_params.index(param) for param in args.params_interest]
    # Load the data

    original_train = np.loadtxt(os.path.join(args.train_sample_dir, "original_train.csv"), delimiter=',')
    original_val = np.loadtxt(os.path.join(args.train_sample_dir, "original_val.csv"), delimiter=',')
    original_test = np.loadtxt(os.path.join(args.train_sample_dir, "original_test.csv"), delimiter=',')
    target_train = np.loadtxt(os.path.join(args.train_sample_dir, "target_train.csv"), delimiter=',')
    target_val = np.loadtxt(os.path.join(args.train_sample_dir, "target_val.csv"), delimiter=',')
    target_test = np.loadtxt(os.path.join(args.train_sample_dir, "target_test.csv"), delimiter=',')

    #we need this for the 3D/8D swd
    original_test_3D = np.loadtxt(os.path.join(args.sample_dir_3D, "original_test.csv"), delimiter=',')
    target_test_3D = np.loadtxt(os.path.join(args.sample_dir_3D, "target_test.csv"), delimiter=',')

    original_test_8D = np.loadtxt(os.path.join(args.sample_dir_8D, "original_test.csv"), delimiter=',')
    target_test_8D = np.loadtxt(os.path.join(args.sample_dir_8D, "target_test.csv"), delimiter=',')

    original_test_all = np.loadtxt(os.path.join(args.sample_dir_all, "original_test.csv"), delimiter=',')
    target_test_all = np.loadtxt(os.path.join(args.sample_dir_all, "target_test.csv"), delimiter=',')


    # Train the model
    
    if args.model == 'binning':
        hyperparams = json.load(open(args.hyperparameters)) if args.hyperparameters is not None else {"n_bins": 12, "n_neighs": 0}
        model = train_binning(original_train, target_train, hyperparams["n_bins"], hyperparams["n_neighs"])
        weights_test = predict_binning(model, original_test)
        logdir = args.logdir + f"/run_{hyperparams['n_bins']}_{hyperparams['n_neighs']}_{int(time.time())}"
        writer = SummaryWriter(logdir)

    elif args.model == 'NN':
        hyperparams = json.load(open(args.hyperparameters)) if args.hyperparameters is not None else {"n_layers": 2, "n_neurons": 15, "epochs": 100, "batch_size": 2048, "activation_function": "tanh"}
        history, model, train_scale_factor = train_NN(original_train, original_val, target_train, target_val, 
                        n_layers = hyperparams["n_layers"], 
                        n_neurons = hyperparams["n_neurons"], 
                        epochs = hyperparams["epochs"], 
                        batch_size = hyperparams["batch_size"], 
                        activation_function = hyperparams["activation_function"])
        weights_test = predict_NN(original_test, model, train_scale_factor)
        logdir = args.logdir + f"/run_{hyperparams['n_layers']}_{hyperparams['n_neurons']}_{hyperparams['epochs']}_{hyperparams['batch_size']}_{hyperparams['activation_function']}_{int(time.time())}"
        writer = SummaryWriter(logdir)

    elif args.model == 'GBR':
        hyperparams = json.load(open(args.hyperparameters)) if args.hyperparameters is not None else {"n_estimators": 110, "learning_rate": 0.01, "max_depth": 6, "min_samples_leaf": 70, "loss_regularization": 4}
        model = train_GBR(original_train, target_train,
                        n_estimators = hyperparams["n_estimators"], 
                        learning_rate = hyperparams["learning_rate"], 
                        max_depth = hyperparams["max_depth"], 
                        min_samples_leaf = hyperparams["min_samples_leaf"], 
                        loss_regularization = hyperparams["loss_regularization"])
        weights_test = predict_GBR(original_test, target_test, model)
        logdir = args.logdir + f"/run_{hyperparams['n_estimators']}_{hyperparams['learning_rate']}_{hyperparams['max_depth']}_{hyperparams['min_samples_leaf']}_{hyperparams['loss_regularization']}_{int(time.time())}"
        writer = SummaryWriter(logdir)
    
    elif args.model == 'XGB':
        hyperparams = json.load(open(args.hyperparameters)) if args.hyperparameters is not None else {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'gamma': 2, 'subsample': 0.3, 'early_stopping_rounds': 10}
        with open(os.path.join(args.train_sample_dir, "original_train.csv")) as f:
                feature_names = f.readline()[1:].strip().split(",")
        model = train_XGB(original_train, original_val, target_train, target_val, hparams = hyperparams, header = feature_names)
        weights_test = predict_XGB(original_test, model, header = feature_names)
        logdir = args.logdir + f"/run_{hyperparams['max_depth']}_{hyperparams['learning_rate']}_{hyperparams['n_estimators']}_{hyperparams['gamma']}_{hyperparams['subsample']}_{hyperparams['early_stopping_rounds']}_{int(time.time())}"
        writer = SummaryWriter(logdir)

    else:
        raise ValueError("Invalid model choice. Please choose from 'binning', 'NN', 'GBR', 'XGB'.")

    weight_dict = {args.model: weights_test}

    dict_mean_swd_3D = compute_swd(original_test_3D, target_test_3D, weight_dict, n_directions = args.n_directions)
    list_swd_3D = np.load(args.swd_distribution_3D)
    p_value_3D = compute_p_value(dict_mean_swd_3D, list_swd_3D)[args.model]

    dict_mean_swd_8D = compute_swd(original_test_8D, target_test_8D, weight_dict, n_directions = args.n_directions)
    list_swd_8D = np.load(args.swd_distribution_8D)
    p_value_8D = compute_p_value(dict_mean_swd_8D, list_swd_8D)[args.model]

    dict_mean_swd_all = compute_swd(original_test_all, target_test_all, weight_dict, n_directions = args.n_directions)
    list_swd_all = np.load(args.swd_distribution_all)
    p_value_all = compute_p_value(dict_mean_swd_all, list_swd_all)[args.model]

    dict_mean_swd_ndim = compute_swd(original_test, target_test, weight_dict, n_directions = args.n_directions)
    list_swd_ndim = np.load(args.custom_swd_distribution)
    p_value_ndim = compute_p_value(dict_mean_swd_ndim, list_swd_ndim)[args.model]


    swd_3d = dict_mean_swd_3D[args.model]
    swd_8d = dict_mean_swd_8D[args.model]
    swd_all = dict_mean_swd_all[args.model]
    swd_ndim = dict_mean_swd_ndim[args.model]

    if args.model == 'binning':
        metrics = {
            "SWD_3D": swd_3d,
            "p_value_3D": p_value_3D,
            "SWD_8D": swd_8d,
            "p_value_8D": p_value_8D,
            "SWD_all": swd_all,
            "p_value_all": p_value_all,
            f"SWD_{original_test.shape[1]}D": swd_ndim,
            f"p_value_{original_test.shape[1]}D": p_value_ndim
        }

    elif args.model == 'NN':
        if len(history.history['val_loss']) < 100:
            best_epoch = np.argmin(history.history['val_loss'])
        else:
            best_epoch = 99
        metrics = {
            "val_loss": history.history['val_loss'][best_epoch],
            "val_AUC": history.history['val_auc'][best_epoch],
            "loss": history.history['loss'][best_epoch],
            "AUC": history.history['auc'][best_epoch],
            "SWD_3D": swd_3d,
            "p_value_3D": p_value_3D,
            "SWD_8D": swd_8d,
            "p_value_8D": p_value_8D,
            "SWD_all": swd_all,
            "p_value_all": p_value_all,
            f"SWD_{original_test.shape[1]}D": swd_ndim,
            f"p_value_{original_test.shape[1]}D": p_value_ndim
        }

    elif args.model == 'GBR':
        metrics = {
            "SWD_3D": swd_3d,
            "p_value_3D": p_value_3D,
            "SWD_8D": swd_8d,
            "p_value_8D": p_value_8D,
            "SWD_all": swd_all,
            "p_value_all": p_value_all,
            f"SWD_{original_test.shape[1]}D": swd_ndim,
            f"p_value_{original_test.shape[1]}D": p_value_ndim
        }

    else:
        best_iter = model.best_iteration
        metrics = {
            "val_logloss": model.evals_result()['validation_1']['logloss'][best_iter],
            "val_auc": model.evals_result()['validation_1']['auc'][best_iter],
            "train_logloss": model.evals_result()['validation_0']['logloss'][best_iter],
            "train_auc": model.evals_result()['validation_0']['auc'][best_iter],
            # "test_logloss": model.evals_result()['validation_2']['logloss'][best_iter],
            # "test_auc": model.evals_result()['validation_2']['auc'][best_iter],
            "SWD_3D": swd_3d,
            "p_value_3D": p_value_3D,
            "SWD_8D": swd_8d,
            "p_value_8D": p_value_8D,
            "SWD_all": swd_all,
            "p_value_all": p_value_all,
            f"SWD_{original_test.shape[1]}D": swd_ndim,
            f"p_value_{original_test.shape[1]}D": p_value_ndim
        }

    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    for i in range(original_test_all.shape[1]):
        if args.analysis_params[i] in binning_dict:
            x_min = binning_dict[args.analysis_params[i]]["x_min"]
            x_max = binning_dict[args.analysis_params[i]]["x_max"]
            n_bins = binning_dict[args.analysis_params[i]]["n_bins"]
        else:
            x_min = None
            x_max = None
            n_bins = 30
        chi2, dof = chi2_hist_axis(original_test_all, target_test_all, weight_dict[args.model], axis_number = i, n_bins=n_bins, x_min = x_min, x_max = x_max)
        metrics[f"chi2_dof_{args.analysis_params[i]}"] = chi2/dof if dof > 0 else 0

    metrics[f"chi2_dof_3D"] = chi2_dof(original_test_3D, target_test_3D, weight_dict, binning_dict=binning_dict, List_param_interest = args.params_3D)[args.model]
    metrics[f"chi2_dof_8D"] = chi2_dof(original_test_8D, target_test_8D, weight_dict, binning_dict=binning_dict, List_param_interest = args.params_8D)[args.model]
    metrics["chi2_dof_all"] = chi2_dof(original_test_all, target_test_all, weight_dict, binning_dict=binning_dict, List_param_interest = args.analysis_params)[args.model]
    metrics[f"chi2_dof_{original_test.shape[1]}D"] = chi2_dof(original_test, target_test, weight_dict, binning_dict=binning_dict, List_param_interest = [args.analysis_params[i] for i in Index_params_interest])[args.model]

    print(f"p_value_3D : {p_value_3D}")
    print(f"p_value_8D : {p_value_8D}")
    print(f"p_value_all : {p_value_all}")
    print(f"p_value_{original_test.shape[1]}D : {p_value_ndim}")


    writer.add_hparams(hyperparams, metrics)

    hparams_metrics_merged = {**hyperparams, **metrics}
    df_row = pd.DataFrame([hparams_metrics_merged])
    csv_path = os.path.join(args.logdir, "Hyperparameters_metrics.csv")
    csv_2_path = args.output_file

    if not os.path.isfile(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)

    if not os.path.isfile(csv_2_path):
        df_row.to_csv(csv_2_path, index=False)
    else:
        df_row.to_csv(csv_2_path, mode='a', header=False, index=False)

    print(f"Run with hyperparameters : {hyperparams} is done.")
