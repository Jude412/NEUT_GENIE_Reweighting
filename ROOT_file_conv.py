"""The objective of this script is to create a function that takes as input a Tree ROOT file and to 
return an array containing the desired parameters of interest. They are to be given as a list containing strings."""

import numpy as np
import uproot
import awkward as ak

def convert_input_file(input_file, input_tree, branches, analysis_params, modes = None, modes_v2 = None):
    """This function takes as input a ROOT FlatTree from and returns an array containing all the parameters of interest.
    They are : E_nu, E_lep, cos(theta_lep), Q2, q0, q3, W, Eav, y, neutrino PDG, interaction, mode, charged current,
    hit nucleus atomic number, hit nucleon PDG, multiplicity of final state proton, neutron, pions and the sum of their kinetic energies. """
    # Open the ROOT file
    file = uproot.open(input_file)
    tree = file[input_tree].arrays(branches, library="ak")
    
    tree["W"] = tree["W_nuc_rest"]
    # For dealing with modes, we use absolute value to allow common treatment of neutrinos and antineutrinos. No ambiguity arises because separate BDTs are trained for neutrinos and antineutrinos.
    tree["Mode"] = abs(tree["Mode"])

    weird_modes = [11, 12, 13, 17, 31, 32, 33, 34, 38, 39]
    if modes is None:
        modes = np.unique(tree["Mode"])
    if bool(set(modes) & set(weird_modes)):
        print("Warning there are undefined (-999) values in some events for the Q2, q0 and q3 parameters in the modes you are trying to extract." \
        "We removed such events from the tree, but be careful.")
        weird_mask = (tree["Q2"] <= -1) | (tree["q0"] <= -1) | (tree["q3"] <= -1) | (tree["Eav"] <= -1) | (tree["W"] <= -1)
        tree = tree[~weird_mask]

    # we now compute the parameters of interest
    # Plep and PTlep are computed from px py pz using the first lepton in the final state (in pdg branch)
    mask_lepton = (
    (np.abs(tree["pdg"]) == 11) |
    (np.abs(tree["pdg"]) == 12) |
    (np.abs(tree["pdg"]) == 13) |
    (np.abs(tree["pdg"]) == 14)
    )

    px_lep = ak.firsts(tree["px"][mask_lepton])
    py_lep = ak.firsts(tree["py"][mask_lepton])
    pz_lep = ak.firsts(tree["pz"][mask_lepton])

    tree["Plep"] = np.sqrt(px_lep**2 + py_lep**2 + pz_lep**2)
    tree["PTlep"] = np.sqrt(px_lep**2 + py_lep**2)


    # Note that the Hank-generated files have no pdg_vert branch, so the following fields can't be created. 
    # This is fine as we will be using a separate BDT per nucleus

    # # A (this may change, since there might be different nuclei in the same file later on)
    # coherent_mask = tree["Mode"] == 16
    # A_value_string = np.unique(tree[coherent_mask]["pdg_vert"][:, 1])[0] # format : 10LZZZAAAI
    # A_value = int(str(A_value_string)[3:6])
    # tree["A"] = A_value
    # # hitnucleon
    # coherent_like_mask = ((tree["Mode"] == 16) | (tree["Mode"] == 36) | (tree["Mode"] == 2) | (tree["Mode"] == 32))
    # tree["hitnuc"] = ak.where(
    #     coherent_like_mask,
    #     -999,
    #     ak.where(tree["pdg_vert"][:, 1] == 1000080160,
    #              tree["pdg_vert"][:, 2],
    #          tree["pdg_vert"][:, 1])
    #             )


    # cut the tree to the desired modes if specified
    if modes is not None:
        mask = False
        for m in modes:
            mask = mask | (tree["Mode"] == m)

        tree = tree[mask]

    # multiplicity and sum of kinetic energies of final state protons, neutrons and pions
    pdg = tree["pdg"]
    E   = tree["E"]
    px = tree["px"]
    py = tree["py"]
    pz = tree["pz"]

    #Multiplicity
    tree["N_n"]   = ak.sum(pdg == 2112, axis=1)
    tree["N_p"]   = ak.sum(pdg == 2212, axis=1)
    tree["N_pi0"] = ak.sum(pdg == 111, axis=1)
    tree["N_pim"] = ak.sum(pdg == -211, axis=1)
    tree["N_pip"] = ak.sum(pdg == 211, axis=1)

    #Sum of kinetic energy
    tree["E_N"]   = E * (pdg == 2112)
    tree["E_P"]   = E * (pdg == 2212)
    tree["E_pi0"] = E * (pdg == 111)
    tree["E_pim"] = E * (pdg == -211)
    tree["E_pip"] = E * (pdg == 211)

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

    # multiplicity of final state photons 

    tree["N_gamma"] = ak.sum(pdg == 22, axis=1)

    # multiplicity of final state particles that are not protons, neutrons, pions or photons

    tree["N_other"] = ak.sum((pdg != 2112) & (pdg != 2212) & (pdg != 111) & (pdg != -211) & (pdg != 211) & (pdg != 22) & (pdg != 13), axis=1)

    # we create a modev2 parameters that gathers the modes based on the number of pions in the final state
    # We choose the following mapping : 0 for CC0pi, 1 for CC1pipm, 2 for CC1pi0, 3 for CCNpi, 4 for CCOther, 5 for the rest, 6 for CCgamma, 7 for CCOther_QE
    mask_CC0pi = (tree["Mode"] <= 30) & (tree["N_pip"] + tree["N_pim"] + tree["N_pi0"] == 0) & (tree["N_gamma"] == 0) & (tree["N_other"] == 0)
    mask_CC1pipm = (tree["Mode"] <= 30) & (tree["N_pip"] + tree["N_pim"] == 1) & (tree["N_pi0"] == 0) & (tree["N_gamma"] == 0) & (tree["N_other"] == 0)
    mask_CC1pi0 = (tree["Mode"] <= 30) & (tree["N_pip"] + tree["N_pim"] == 0) & (tree["N_pi0"] == 1) & (tree["N_gamma"] == 0) & (tree["N_other"] == 0)
    mask_CCNpi = (tree["Mode"] <= 30) & (tree["N_pip"] + tree["N_pim"] + tree["N_pi0"] >= 2) & (tree["N_gamma"] == 0) & (tree["N_other"] == 0)
    mask_CCgamma = (tree["Mode"] <= 30) & (tree["N_gamma"] >= 1)
    mask_CCOther_QE = (tree["Mode"] == 1) & (~mask_CC0pi) & (~mask_CC1pipm) & (~mask_CC1pi0) & (~mask_CCNpi) & (~mask_CCgamma)
    mask_CCOther = (tree["Mode"] <= 30) & (~mask_CC0pi) & (~mask_CC1pipm) & (~mask_CC1pi0) & (~mask_CCNpi) & (~mask_CCgamma) & (~mask_CCOther_QE)
    tree["Mode_v2"] = ak.where(mask_CC0pi, 0,
                        ak.where(mask_CC1pipm, 1,
                            ak.where(mask_CC1pi0, 2,
                                ak.where(mask_CCNpi, 3,
                                    ak.where(mask_CCgamma, 6,
                                        ak.where(mask_CCOther_QE, 7,
                                            ak.where(mask_CCOther, 4, 5)
                                        ))))))
    
    
    # cut the tree to the desired modes_v2 if desired

    if modes_v2 is not None:
        mask = False
        for m in modes_v2:
            mask = mask | (tree["Mode_v2"] == m)

        tree = tree[mask]

    if 6 not in modes_v2:
        print("Warning : the CCgamma mode is not included. If you want to include it, please add 6 to the modes_v2 list.")

    if 7 not in modes_v2:
        print("Warning : the CCOther_QE mode is not included. If you want to include it, please add 7 to the modes_v2 list.")

    # we now create the final array containing the parameters of interest
    param_values = []
    for param in analysis_params:
        param_values.append(ak.to_numpy(tree[param]))
    data = np.array(np.column_stack(param_values))
    ## Check for NaN
    for i in range(data.shape[1]):
        if np.isnan(data[:, i]).any():
            print(f"Warning: NaN values found in column {i} of the data array.")
            data = data[~np.isnan(data[:, i])].copy()
        else:
            pass
    return data
