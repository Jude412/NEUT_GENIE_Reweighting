"""This script takes as input the csv files containing the splitted samples for training, trains the specified model and
saves the weights predicted by the model for each samples as well as the model itself.
The script uses extensively the functions defined in Train_predict.py"""

#imports
import numpy as np
from Train_predict import train_binning, predict_binning, train_NN, predict_NN, train_GBR, predict_GBR, train_XGB, predict_XGB
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

    original_train = np.loadtxt(args.original_train, delimiter=',')
    original_val = np.loadtxt(args.original_val, delimiter=',')
    original_test = np.loadtxt(args.original_test, delimiter=',')
    target_train = np.loadtxt(args.target_train, delimiter=',')
    target_val = np.loadtxt(args.target_val, delimiter=',')
    target_test = np.loadtxt(args.target_test, delimiter=',')

    for model_name in args.model_list:
        if model_name == 'binning':
            if args.hparams_dict is not None:
                all_hparams = json.load(open(args.hparams_dict))
                hyperparams = all_hparams[model_name]
            else:
                hyperparams = {"n_bins": 12, "n_neighs": 0}
            model = train_binning(original_train, target_train, hyperparams["n_bins"], hyperparams["n_neighs"])
            weights_train = predict_binning(model, original_train)
            weights_val = predict_binning(model, original_val)
            weights_test = predict_binning(model, original_test)

        elif model_name == 'NN':
            if args.hparams_dict is not None:
                all_hparams = json.load(open(args.hparams_dict))
                hyperparams = all_hparams[model_name]
            else:
                hyperparams = {"n_layers": 2, "n_neurons": 15, "epochs": 100, "batch_size": 2048, "activation_function": "tanh"}
            history, model, train_scale_factor = train_NN(original_train, original_val, target_train, target_val, 
                            n_layers = hyperparams["n_layers"], 
                            n_neurons = hyperparams["n_neurons"], 
                            epochs = hyperparams["epochs"], 
                            batch_size = hyperparams["batch_size"], 
                            activation_function = hyperparams["activation_function"])
            weights_train = predict_NN(original_train, model, train_scale_factor)
            weights_val = predict_NN(original_val, model, train_scale_factor)
            weights_test = predict_NN(original_test, model, train_scale_factor)
            # print(type(history.history['loss']), history.history['loss'])
            # print(type(history.history['val_loss']), history.history['val_loss'])
            df = np.concatenate((history.history['loss'], history.history['val_loss']), axis=0)
            np.savetxt(os.path.join(args.save_model_path, f"NN_training_history_{np.shape(original_train)[1]}D.csv"), df, delimiter=',')
        
        elif model_name == 'GBR':
            if args.hparams_dict is not None:
                all_hparams = json.load(open(args.hparams_dict))
                hyperparams = all_hparams[model_name]
            else:
                hyperparams = {"n_estimators": 110, "learning_rate": 0.01, "max_depth": 6, "min_samples_leaf": 70, "loss_regularization": 4}
            model = train_GBR(original_train, target_train,
                            n_estimators = hyperparams["n_estimators"], 
                            learning_rate = hyperparams["learning_rate"], 
                            max_depth = hyperparams["max_depth"], 
                            min_samples_leaf = hyperparams["min_samples_leaf"], 
                            loss_regularization = hyperparams["loss_regularization"])
            weights_train = predict_GBR(original_train, target_train, model)
            weights_val = predict_GBR(original_val, target_val, model)
            weights_test = predict_GBR(original_test, target_test, model)

        elif model_name == 'XGB':
            if args.hparams_dict is not None:
                all_hparams = json.load(open(args.hparams_dict))
                hyperparams = all_hparams[model_name]
            else:
                hyperparams = {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'gamma': 2, 'subsample': 0.3, 'early_stopping_rounds': 10}
            with open(args.original_train) as f:
                feature_names = f.readline()[1:].strip().split(",")
            model = train_XGB(original_train, original_val, target_train, target_val, hparams = hyperparams, header = feature_names)
            weights_train = predict_XGB(original_train, model, header = feature_names)
            weights_val = predict_XGB(original_val, model, header = feature_names)
            weights_test = predict_XGB(original_test, model, header = feature_names)

        else:
            raise ValueError("Invalid model choice. Please choose from 'binning', 'NN', 'GBR', 'XGB'.")
        
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
        


        
