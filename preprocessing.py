import numpy as np

max_PRI_jet_num = 4

columns = {
    "DER_mass_MMC": 0,
    "DER_mass_transverse_met_lep": 1,
    "DER_mass_vis": 2,
    "DER_pt_h": 3,
    "DER_deltaeta_jet_jet": 4,
    "DER_mass_jet_jet": 5,
    "DER_prodeta_jet_jet": 6,
    "DER_deltar_tau_lep": 7,
    "DER_pt_tot": 8,
    "DER_sum_pt": 9,
    "DER_pt_ratio_lep_tau": 10,
    "DER_met_phi_centrality": 11,
    "DER_lep_eta_centrality": 12,
    "PRI_tau_pt": 13,
    "PRI_tau_eta": 14,
    "PRI_tau_phi": 15,
    "PRI_lep_pt": 16,
    "PRI_lep_eta": 17,
    "PRI_lep_phi": 18,
    "PRI_met": 19,
    "PRI_met_phi": 20,
    "PRI_met_sumet": 21,
    "PRI_jet_num": 22,
    "PRI_jet_leading_pt": 23,
    "PRI_jet_leading_eta": 24,
    "PRI_jet_leading_phi": 25,
    "PRI_jet_subleading_pt": 26,
    "PRI_jet_subleading_eta": 27,
    "PRI_jet_subleading_phi": 28,
    "PRI_jet_all_pt": 29
}


PRI_jet_num_columns = {
    0: [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltar_tau_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_tau",
        "DER_met_phi_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_met_sumet"
    ],
    1: [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltar_tau_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_tau",
        "DER_met_phi_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_met_sumet",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_all_pt"
    ],
    2: [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_deltar_tau_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_tau",
        "DER_met_phi_centrality",
        "DER_lep_eta_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_met_sumet",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_jet_all_pt"
    ],
    3: [
        "DER_mass_MMC",
        "DER_mass_transverse_met_lep",
        "DER_mass_vis",
        "DER_pt_h",
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_deltar_tau_lep",
        "DER_pt_tot",
        "DER_sum_pt",
        "DER_pt_ratio_lep_tau",
        "DER_met_phi_centrality",
        "DER_lep_eta_centrality",
        "PRI_tau_pt",
        "PRI_tau_eta",
        "PRI_tau_phi",
        "PRI_lep_pt",
        "PRI_lep_eta",
        "PRI_lep_phi",
        "PRI_met",
        "PRI_met_phi",
        "PRI_met_sumet",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_jet_all_pt"
    ]
}


def DER_mass_MMC_splitt(y, X, ids):
    r = X[:, columns["DER_mass_MMC"]] != -999

    return (y[r], X[r], ids[r]), (y[~r], X[~r, 1:], ids[~r])


def PRI_jet_num_split(y, X, ids):
    res = []

    for PRI_jet_num in range(max_PRI_jet_num):
        r = X[:, columns["PRI_jet_num"]] == PRI_jet_num
        c = [ columns[column] for column in PRI_jet_num_columns[PRI_jet_num] ]

        set1, set2 = DER_mass_MMC_splitt(y[r], X[r][:, c], ids[r])

        res.append(set1)
        res.append(set2)

    return res


def standardize(X_train, X_test):
    mean = np.mean(X_train)
    X_train = X_train - mean
    X_test = X_test - mean

    std = np.std(X_train)
    X_train = X_train / std
    X_test = X_test / std

    return X_train, X_test
