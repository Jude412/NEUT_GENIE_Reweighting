"""The goal of this script is train and save a reweighting model for one model and set of hyperparameters."""

#imports
from Sample_io import load_samples
from Train_predict import hyperparameters_of, predict_model, train_model, save_model
import os
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a reweighting model with one set of hyperparameters and evaluate it.")
    argparser.add_argument('--sample_dir', type=str, required=True,
                           help='Directory where the training original and target samples are stored in parquet format.')
    argparser.add_argument('--model', type=str, required=True, help='The reweighting model to train.')
    argparser.add_argument('--topology', type=str, required=True, help='The topology to filter the training samples on.')
    argparser.add_argument('--hyperparameters', type=str, required=True, help="Path to the JSON file containing the hyperparameters to train the model with.")
    argparser.add_argument("--reweighting_params", nargs="+", required=True, help="List of parameters to use for the reweighting.")
    argparser.add_argument("--output_dir", type=str, required=True, help="The path to the output directory for the model.")
    args = argparser.parse_args()

    # Load the training and validation samples for the reweighting parameter set
    training_samples = load_samples(args.sample_dir, topology=args.topology, params=args.reweighting_params, sample_names=("original_train", "target_train", "original_val", "target_val"))

    # Train the model
    hyperparams = hyperparameters_of(args.model, args.hyperparameters)
    model = train_model(args.model, training_samples, hyperparams)
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{args.model}")
    save_model(args.model, model, model_path)
    print(f"{args.model} model saved at {model_path}")
