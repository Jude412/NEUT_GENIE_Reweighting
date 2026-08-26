"""The goal of this script is to build the samples holding only the parameters the reweighting models
are trained on.

The training, validation and test samples holding every analysis parameter are created once by 'Init.py',
which also splits the events. This script simply keeps the columns of the parameters of interest, so that
the events of a given sample are exactly the ones 'Init.py' put in it: the split is never recomputed and
the input files are never read a second time. The samples are saved as csv files in a specified directory.
The script can be run from the command line with the appropriate arguments."""

#imports 
from Sample_io import load_sample, save_sample, sample_columns
import argparse
import os

# The samples created by 'Init.py' that are reduced to the parameters of interest.
SAMPLE_NAMES = ("original_train", "original_val", "original_test",
                "target_train", "target_val", "target_test")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to create the training and validation samples for the reweighting techniques.")
    argparser.add_argument("--init_samples_dir", required=True, type=str,
                           help="Path to the directory holding the samples of all the analysis parameters created by Init.py.")
    argparser.add_argument("--samples_dir", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/vols/dune/jmm224/t2knova/reweighting/saved_samples/first_test/")
    argparser.add_argument("--parameters_interest", nargs='+', required=False, help="List of parameters to keep from the original files, in the format 'param1,param2,...'.",
                           default=["Enu_true", "ELep", "CosLep", "W", "Eav"])
    args = argparser.parse_args()

    # Keeping only the parameters of interest of the samples created by Init.py
    print("Keeping the parameters of interest of the training, validation and test samples...")

    os.makedirs(args.samples_dir, exist_ok=True)
    for sample_name in SAMPLE_NAMES:
        input_file = os.path.join(args.init_samples_dir, f"{sample_name}.csv")
        distribution, weights = load_sample(input_file)

        columns = sample_columns(input_file)
        missing_params = [param for param in args.parameters_interest if param not in columns]
        if missing_params:
            raise ValueError(
                f"The parameters {missing_params} are not held by {input_file}, which holds {columns}. "
                "Every parameter of the 'parameters'/'reweighting' section of the config file must also be "
                "listed in its 'parameters'/'all' section."
            )

        index_params = [columns.index(param) for param in args.parameters_interest]
        save_sample(os.path.join(args.samples_dir, f"{sample_name}.csv"),
                    distribution[:, index_params], weights, args.parameters_interest)
