"""This script takes as input the csv files containing the splitted samples for training, trains the specified models and
saves the weights predicted by each model for every sample as well as the model itself.
The script uses extensively the functions defined in Train_predict.py.

The paths of the weights and of the model of every trained model are gathered in a single json file, so that
the scripts computing the metrics and making the plots never have to guess them from one another."""

#imports
import numpy as np
from Sample_io import load_samples
from Train_predict import hyperparameters_of, predict_model, train_model
import argparse
import pickle
import json
import os

# The samples the weights of a model are predicted for.
PREDICTED_SAMPLES = ("original_train", "original_val", "original_test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the reweighting models and save the weights and the models themselves.')
    parser.add_argument('--sample_dir', type=str, required=False, help='Directory holding the training, validation and test samples as csv files.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/4D_Eav/")
    parser.add_argument('--model_list', required=True, nargs='+', help='List of the reweighting models to train.')
    parser.add_argument('--hparams_dict', type=str, default=None, help='Path to the file containing the dictionary of hyperparameters for the models in json format. If not provided, default hyperparameters will be used.')
    parser.add_argument('--save_weights_path', type=str, required=False, help='Path to save the predicted weights csv file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_weights/4D_Eav/")
    parser.add_argument('--save_model_path', type=str, required=False, help='Path to save the trained model file.',
                        default="/vols/dune/jmm224/t2knova/reweighting/saved_models/4D_Eav/")
    parser.add_argument('--save_path_dict', '--save_weight_path_dict', dest='save_path_dict', type=str, required=True,
                        help="Path to the json file where the paths to the weights and to the model of every trained model will be saved.")
    args = parser.parse_args()

    samples = load_samples(args.sample_dir)
    n_params = samples["original_train"][0].shape[1]

    os.makedirs(args.save_weights_path, exist_ok=True)
    os.makedirs(args.save_model_path, exist_ok=True)

    saved_paths = {}
    for model_name in args.model_list:
        hyperparams = hyperparameters_of(model_name, args.hparams_dict)
        model = train_model(model_name, samples, hyperparams)

        for sample_name in PREDICTED_SAMPLES:
            weights = predict_model(model_name, model, samples[sample_name][0])
            # The name of the sample the weights belong to is kept as it was, so that
            # 'original_test' gives the historical 'weights_test' file name.
            split = sample_name.split("_")[-1]
            np.savetxt(os.path.join(args.save_weights_path, f"{model_name}_weights_{split}_{n_params}D.csv"),
                       weights, delimiter=',')

        model_file = os.path.join(args.save_model_path, f"{model_name}_model_{n_params}D.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        saved_paths[model_name] = {
            "weights": os.path.join(args.save_weights_path, f"{model_name}_weights_test_{n_params}D.csv"),
            "model": model_file,
        }

        print(f"Model {model_name} trained and saved successfully. Weights for train, val and test sets saved in {args.save_weights_path}. Model saved in {args.save_model_path}. History of training (if applicable) saved in {args.save_model_path} as well.")

    with open(args.save_path_dict, 'w') as f:
        json.dump(saved_paths, f, indent=0)
