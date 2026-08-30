"""The objective of this script is to create a function that takes as input a Tree ROOT file and to 
return an array containing the desired parameters of interest. They are to be given as a list containing strings."""

import numpy as np
import uproot
import awkward as ak
from typing import cast
from Sample_io import WEIGHT_COLUMN

# Branches to read in
BRANCHES = ["Enu_true", "ELep", "PLep", "CosLep", "Eav", "Q2", "q0", "q3", "W_nuc_rest", "y", "PDGnu", "Mode", "cc", "nfsp", "px", "py", "pz", "E", "pdg", "px", "py", "pz", "PDGLep", "fScaleFactor", "RWWeight"]
# Branches whose product gives the total weight of an event.
WEIGHT_BRANCHES = ("RWWeight", "fScaleFactor")

# Mapping between the topology names (used in the config file and as the {topology} wildcard
# of the workflow) and the integer values stored in the "Topology" parameter.
TOPOLOGY_CODES = {
    "CC0pi": 0,
    "CC1pipm": 1,
    "CC1pi0": 2,
    "CCNpi": 3,
    "CCOther": 4,
    "NC0pi": 5,
    "NC1pipm": 6,
    "NC1pi0": 7,
    "NCNpi": 8,
    "NCOther": 9,
    "Other": 10,
}

def topology_code(topology):
    """Return the integer code of a topology given either its name (ex: 'CC1pipm') or its code (ex: 1)."""
    if isinstance(topology, str):
        if topology in TOPOLOGY_CODES:
            return TOPOLOGY_CODES[topology]
        try:
            topology = int(topology)
        except ValueError:
            raise ValueError(f"Unknown topology '{topology}'. Please choose from {list(TOPOLOGY_CODES)}.")
    if topology not in TOPOLOGY_CODES.values():
        raise ValueError(f"Unknown topology '{topology}'. Please choose from {list(TOPOLOGY_CODES)}.")
    return int(topology)

def convert_input_file(input_file, input_tree, analysis_params, modes = None, topologies = None, return_weights = False):
    """This function takes as input a ROOT FlatTree from and returns an array containing all the parameters of interest."""
    # Open the ROOT file
    file = uproot.open(input_file)
    tree = cast(uproot.TTree, file[input_tree]).arrays(BRANCHES, library="ak")
    
    tree["W"] = tree["W_nuc_rest"]

    # Per-event weight: the total weight of an event is the product of its reweighting weight and of its scale factor.
    if WEIGHT_BRANCHES[0] in tree.fields and WEIGHT_BRANCHES[1] in tree.fields:
        tree[WEIGHT_COLUMN] = tree[WEIGHT_BRANCHES[0]] * tree[WEIGHT_BRANCHES[1]]
    else:
        print(f"Warning: the branches {WEIGHT_BRANCHES} were not both read from {input_file}: "
              "every event is given a weight of 1. Add them to BRANCHES")
        tree[WEIGHT_COLUMN] = ak.ones_like(tree["W"])

    # For dealing with modes, we use absolute value to allow common treatment of neutrinos and antineutrinos. No ambiguity arises because separate BDTs are trained for neutrinos and antineutrinos.
    tree["Mode"] = abs(tree["Mode"])
    # cut the tree to the desired modes if specified
    if modes is not None:
        mask = False
        for m in modes:
            mask = mask | (tree["Mode"] == m)

        tree = tree[mask]

    # we now compute the parameters of interest
    # PLep and PTlep are computed from px py pz using the first lepton in the final state (in pdg branch)
    tree["PTlep"] = tree["PLep"]*np.sqrt(1 - (tree["CosLep"]**2))

    # multiplicity and sum of kinetic energies of final state protons, neutrons and pions
    pdg = tree["pdg"]
    energy = tree["E"]
    px = tree["px"]
    py = tree["py"]
    pz = tree["pz"]

    #Multiplicity
    tree["N_n"]   = ak.sum(pdg == 2112, axis=1)
    tree["N_p"]   = ak.sum(pdg == 2212, axis=1)
    tree["N_pi0"] = ak.sum(pdg == 111, axis=1)
    tree["N_pim"] = ak.sum(pdg == -211, axis=1)
    tree["N_pip"] = ak.sum(pdg == 211, axis=1)
    tree["N_gamma"] = ak.sum((pdg == 22), axis=1)
    tree["N_other"] = ak.sum((pdg != 2112) & (pdg != 2212) & (pdg != 111) & (pdg != 211) & (pdg != -211) & (pdg != 22) & (pdg < 1000000000), axis=1) - 1 # -1 to remove prim lepton

    #Sum of kinetic energy
    tree["E_N"]   = energy * (pdg == 2112)
    tree["E_P"]   = energy * (pdg == 2212)
    tree["E_pi0"] = energy * (pdg == 111)
    tree["E_pim"] = energy * (pdg == -211)
    tree["E_pip"] = energy * (pdg == 211)

    tree["P2_N"] = px**2 * (pdg == 2112) + py**2 * (pdg == 2112) + pz**2 * (pdg == 2112)
    tree["P2_P"] = px**2 * (pdg == 2212) + py**2 * (pdg == 2212) + pz**2 * (pdg == 2212)
    tree["P2_pi0"] = px**2 * (pdg == 111) + py**2 * (pdg == 111) + pz**2 * (pdg == 111)
    tree["P2_pim"] = px**2 * (pdg == -211) + py**2 * (pdg == -211) + pz**2 * (pdg == -211)
    tree["P2_pip"] = px**2 * (pdg == 211) + py**2 * (pdg == 211) + pz**2 * (pdg == 211)

    tree["K_n"]   = ak.sum(tree["E_N"] - np.sqrt(tree["E_N"]**2 - tree["P2_N"]), axis=1)
    tree["K_p"]   = ak.sum(tree["E_P"] - np.sqrt(tree["E_P"]**2 - tree["P2_P"]), axis=1)
    tree["K_pi0"] = ak.sum(tree["E_pi0"] - np.sqrt(tree["E_pi0"]**2 - tree["P2_pi0"]), axis=1) 
    tree["K_pim"] = ak.sum(tree["E_pim"] - np.sqrt(tree["E_pim"]**2 - tree["P2_pim"]), axis=1)
    tree["K_pip"] = ak.sum(tree["E_pip"] - np.sqrt(tree["E_pip"]**2 - tree["P2_pip"]), axis=1)
    tree["E_gamma"] = ak.sum(energy * (pdg == 22), axis=1)


    # we create a topology parameter that gathers the modes based on the number of pions in the final state

    # Previous analysis omitted photons above 10 MeV from _C_pi topologies
    gamma_deexcite_cut = 10e-3 # If E_gamma < 10 MeV, we consider it a de-excitation photon and don't count it towards the multiplicity
    clean = (tree["E_gamma"] < gamma_deexcite_cut) & (tree["N_other"] == 0)  # no photons/other particles
    # clean = (tree["N_other"] == 0)  # other option: nothing but prim lep, nucleons, pions, and photons in final state

    is_cc = tree["cc"]
    n_pi_charged = tree["N_pip"] + tree["N_pim"]
    n_pi_total = n_pi_charged + tree["N_pi0"]

    topology_masks = {}
    for sign, cc_mask in [("CC", is_cc), ("NC", ~is_cc)]:
        topology_masks[f"{sign}0pi"]   = cc_mask & (n_pi_total == 0) & clean
        topology_masks[f"{sign}1pipm"] = cc_mask & (n_pi_charged == 1) & (tree["N_pi0"] == 0) & clean
        topology_masks[f"{sign}1pi0"]  = cc_mask & (n_pi_charged == 0) & (tree["N_pi0"] == 1) & clean
        topology_masks[f"{sign}Npi"]   = cc_mask & (n_pi_total >= 2) & clean
        # "Other" = everything else in this CC/NC branch (including non-clean final states)
        specific = (
            topology_masks[f"{sign}0pi"] | topology_masks[f"{sign}1pipm"]
            | topology_masks[f"{sign}1pi0"] | topology_masks[f"{sign}Npi"]
        )
        topology_masks[f"{sign}Other"] = cc_mask & ~specific

    # sanity check: every event should match exactly one topology
    mask_sum = ak.sum([ak.values_astype(m, np.int64) for m in topology_masks.values()], axis=0)
    assert ak.all(mask_sum == 1), (
        f"Topology masks are not mutually exclusive/exhaustive: "
        f"{ak.sum(mask_sum == 0)} events matched none, "
        f"{ak.sum(mask_sum > 1)} events matched multiple."
    )

    tree["Topology"] = ak.full_like(tree["Mode"], topology_code("Other"), dtype=np.int64)  # sensible default
    for top in topology_masks.keys():
        tree["Topology"] = ak.where(topology_masks[top], topology_code(top), tree["Topology"])

    # We now cut the tree to the desired topologies if specified
    if topologies is not None:
        mask = False
        for top in topologies:
            mask = mask | (tree["Topology"] == topology_code(top))

        tree = tree[mask]

    # we now create the final array containing the parameters of interest
    # The weights are carried along as an extra column, so that the NaN cleaning below removes the same
    # events from the parameters and from the weights.
    columns = list(analysis_params) + ([WEIGHT_COLUMN] if return_weights else [])
    param_values = []
    for param in columns:
        param_values.append(ak.to_numpy(tree[param]))
    data = np.array(np.column_stack(param_values))
    ## Check for NaN
    for i in range(data.shape[1]):
        if np.isnan(data[:, i]).any():
            print(f"Warning: NaN values found in column {i} of the data array.")
            data = data[~np.isnan(data[:, i])].copy()
        else:
            pass
    if return_weights:
        return data[:, :-1], data[:, -1]
    return data
