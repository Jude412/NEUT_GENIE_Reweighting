"""This script takes as input the csv files containing the splitted samples for training, trains the specified model and
saves the weights predicted by the model for each samples as well as the model itself.
The script uses extensively the functions defined in Train_predict.py"""

#imports
import numpy as np
from Sample_io import load_sample, sample_columns
from Train_predict import train_binning, predict_binning, train_XGB, predict_XGB
import argparse
import pickle
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a reweighting model and save the weights and the model itself.')
    parser.add_argument('--original_train', type=str, required=False, help='Path to the original training sample csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/4D_Eav/original_train.csv")
    parser.add_argument('--original_val', type=str, required=False, help='Path to the original validation sample csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/4D_Eav/original_val.csv")
    parser.add_argument('--original_test', type=str, required=False, help='Path to the original test sample csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/4D_Eav/original_test.csv")
    parser.add_argument('--target_train', type=str, required=False, help='Path to the target training sample csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/4D_Eav/target_train.csv")
    parser.add_argument('--target_val', type=str, required=False, help='Path to the target validation sample csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/4D_Eav/target_val.csv")
    parser.add_argument('--target_test', type=str, required=False, help='Path to the target test sample csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/4D_Eav/target_test.csv")
    parser.add_argument('--model_list', required=True, nargs='+', help='List of the reweighting models to train.')
    parser.add_argument('--hparams_dict', type=str, default=None, help='Path to the file containing the dictionary of hyperparameters for the models in json format. If not provided, default hyperparameters will be used.')
    parser.add_argument('--save_weights_path', type=str, required=False, help='Path to save the predicted weights csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_weights/4D_Eav/")
    parser.add_argument('--save_model_path', type=str, required=False, help='Path to save the trained model file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_models/4D_Eav/")
    parser.add_argument('--save_weight_path_dict', type=str, help = "path to the json file where the dictionary " \
    "   containing the paths to the predicted weights for each model will be saved.")
    args = parser.parse_args()

    original_train, original_train_weight = load_sample(args.original_train)
    original_val, original_val_weight = load_sample(args.original_val)
    original_test, original_test_weight = load_sample(args.original_test)
    target_train, target_train_weight = load_sample(args.target_train)
    target_val, target_val_weight = load_sample(args.target_val)
    target_test, target_test_weight = load_sample(args.target_test)

    for model_name in args.model_list:
        if model_name == 'binning':
            if args.hparams_dict is not None:
                all_hparams = json.load(open(args.hparams_dict))
                hyperparams = all_hparams[model_name]
            else:
                hyperparams = {"n_bins": 12, "n_neighs": 0}

            model = train_binning(original_train, target_train, hyperparams["n_bins"], hyperparams["n_neighs"], original_train_weight=original_train_weight, target_train_weight=target_train_weight)
            weights_train = predict_binning(model, original_train, original_weight=original_train_weight)
            weights_val = predict_binning(model, original_val, original_weight=original_val_weight)
            weights_test = predict_binning(model, original_test, original_weight=original_test_weight)

        elif model_name == 'XGB':
            if args.hparams_dict is not None:
                all_hparams = json.load(open(args.hparams_dict))
                hyperparams = all_hparams[model_name]
            else:
                hyperparams = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'gamma': 2, 'subsample': 0.3, 'early_stopping_rounds': 10}

            model = train_XGB(original_train, original_val, target_train, target_val, hparams = hyperparams,
                              original_train_weight=original_train_weight, original_val_weight=original_val_weight,
                              target_train_weight=target_train_weight, target_val_weight=target_val_weight)
            weights_train = predict_XGB(original_train, model)
            weights_val = predict_XGB(original_val, model)
            weights_test = predict_XGB(original_test, model)

        else:
            raise ValueError("Invalid model choice. Please choose from 'binning', 'XGB'.")
        
        os.makedirs(args.save_weights_path, exist_ok=True)
        os.makedirs(args.save_model_path, exist_ok=True)
        np.savetxt(os.path.join(args.save_weights_path, f"{model_name}_weights_train_{np.shape(original_train)[1]}D.csv"), weights_train, delimiter=',')
        np.savetxt(os.path.join(args.save_weights_path, f"{model_name}_weights_val_{np.shape(original_val)[1]}D.csv"), weights_val, delimiter=',')
        np.savetxt(os.path.join(args.save_weights_path, f"{model_name}_weights_test_{np.shape(original_test)[1]}D.csv"), weights_test, delimiter=',')
        with open(os.path.join(args.save_model_path, f"{model_name}_model_{np.shape(original_train)[1]}D.pkl"), 'wb') as f:
            pickle.dump(model, f)

        print(f"Model {model_name} trained and saved successfully. Weights for train, val and test sets saved in {args.save_weights_path}. Model saved in {args.save_model_path}. History of training (if applicable) saved in {args.save_model_path} as well.")

    weights_path_dict ={}
    for model_name in args.model_list:
        weights_path_dict[f"{model_name}"] = os.path.join(args.save_weights_path, f"{model_name}_weights_test_{np.shape(original_test)[1]}D.csv")
    
    json.dump(weights_path_dict, open(os.path.join(args.save_weight_path_dict), 'w'), indent=0)
        


        
