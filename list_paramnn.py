import json
import os
import argparse

args = argparse.ArgumentParser(description='Generate json files for each NN hyperparameter combination.')
args.add_argument('--output_dir', type=str, default='hps', help='Directory to save the json files.')
args = args.parse_args()

layers_list = [2, 4, 6]
nneurons_list = [4, 6, 8, 10, 12]
epochs_list = [100, 200]
batch_size_list = [512, 1024, 2048, 4096]
activation_function_list = ["relu", "tanh"]

sub_file_template = ""

counter = 0

os.makedirs(os.path.join(args.output_dir, "NN"), exist_ok=True)

for nlayer in layers_list:
    for nneuron in nneurons_list:
        for epoch in epochs_list:
            for batch_size in batch_size_list:
                for activation_function in activation_function_list:
                    with open(os.path.join(args.output_dir, f"NN/NN_hp_{counter}.json"), "w") as f:
                        json.dump({"n_layers": nlayer, "n_neurons": nneuron, "epochs": epoch, "batch_size": batch_size, "activation_function": activation_function}, f)

                    sub_file_template += os.path.join("/vols/dune/jmm224/t2knova/reweighting/", args.output_dir, "NN/NN_hp_{counter}.json") + "\n"
                    counter += 1

sub_file_template += ")"

with open(os.path.join(args.output_dir, "NN/NN_job.sub"), "w") as f:
    f.write(sub_file_template)
