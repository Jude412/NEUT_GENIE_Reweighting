import json
import os

n_bins = [5, 6, 7, 8, 9, 10, 11, 12]
n_neighs = [0, 1, 2]

sub_file_template = ""

counter = 0

os.makedirs("hps/binning", exist_ok=True)

for n_bin in n_bins:
    for n_neigh in n_neighs:
        with open(f"hps/binning/binning_hp_{counter}.json", "w") as f:
            json.dump({"n_bins": n_bin, "n_neighs": n_neigh}, f)    
        sub_file_template += f"/vols/dune/jmm224/t2knova/reweighting/hps/binning/binning_hp_{counter}.json\n"
        counter += 1

sub_file_template += ")"

with open("hps/binning/binning_job.sub", "w") as f:
  f.write(sub_file_template)
