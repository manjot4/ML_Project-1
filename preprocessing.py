import numpy as np

# All features names
features = {
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

# Features that particular "PRI_jet_num" measures
PRI_jet_num_features = {
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


def __DER_mass_MMC_split(y, tx, ids):
    """ Splits a "PRI_jet_num" subset into two subsets. One contains the "DER_mass_MMC" feature
    with defined values and in the other the "DER_mass_MMC" feature is dropped.

    Parameters
    ---------
    y: ndarray
        The target variables
    tx: ndarray
        The dataset
    ids: ndarray
        The ids

    Returns
    -------
    tuple
        The two subsets
    """

    # Find the rows in tx where the "DER_mass_MMC" feature has the
    # defined values.
    r = tx[:, features["DER_mass_MMC"]] != -999

    # Create the first subset extracting the rows with the defined
    # "DER_mass_MMC" feature values. 
    ds_1 = (y[r], tx[r], ids[r])

    # Create the second subset extracting the rows with the undefined
    # "DER_mass_MMC" feature values and drop the "DER_mass_MMC" feature.
    ds_2 = (y[~r], tx[~r, 1:], ids[~r])

    return ds_1, ds_2


def __PRI_jet_num_split(y, tx, ids, PRI_jet_num_vals):
    """ Splits a dataset using the "PRI_jet_num" feature removing the
    unnecessary nan values.

    Parameters
    ----------
    y: ndarray
        The target variables
    tx: ndarray
        The dataset
    ids: ndarray
        The ids
    PRI_jet_num_vals:
        The "PRI_jet_num" values

    Returns
    -------
    array
        Array of tuples each containing subsets of y, tx, and ids
    """
    res = []

    # Iterate over each possible value for "PRI_jet_num".
    for PRI_jet_num in PRI_jet_num_vals:
        
        # Find the rows in tx where the "PRI_jet_num" feature has the current
        # PRI_jet_num value.
        r = tx[:, features["PRI_jet_num"]] == PRI_jet_num
        
        # Extract the features measured by the current PRI_jet_num. Discard
        # the others since they only contain nan values.
        f = [ features[feature] for feature in PRI_jet_num_features[PRI_jet_num] ]

        # Split the dataset further since the "DER_mass_MMC" feature has
        # too much nan values.
        ds_1, ds_2 = __DER_mass_MMC_split(y[r], tx[r][:, f], ids[r])

        # Append the newly created datasets.
        res.append(ds_1)
        res.append(ds_2)

    return res


def PRI_jet_num_split(y, tx, ids, combine_vals=False):
    """ Wrapper around "__PRI_jet_num_split" that does the actual split.

    Parameters
    ----------
    y: ndarray
        The target variables
    tx: ndarray
        The dataset
    ids: ndarray
        The ids
    combine_vals: boolean
        Whether to combine values 2 and 3.

    Returns
    -------
    array
        Array of tuples each containing subsets of y, tx, and ids
    """

    # Copy the sets so that the original data is preserved.
    y_cpy = np.copy(y)
    X_cpy = np.copy(tx)
    ids_cpy = np.copy(ids)

    # Combine values 2 and 3 by setting the value 3 to 2.
    if combine_vals:
        f = X_cpy[:, features["PRI_jet_num"]]
        f = np.where(f < 3, f, 2)
        X_cpy[:, features["PRI_jet_num"]] = f

        return __PRI_jet_num_split(y_cpy, X_cpy, ids_cpy, [0, 1, 2])

    else:
        return __PRI_jet_num_split(y_cpy, X_cpy, ids_cpy, [0, 1, 2, 3])


def standardize(X_train, X_test):
    """ Standardizes the train and test datasets using the mean and
    stadard deviation vectors from the train dataset.

    Parameters
    ----------
    X_train: ndarray
        The train dataset
    X_test: ndarray
        The test dataset

    Returns
    -------
    touple
        Standardized train and test datasets
    """
    # Copy the sets so that the original data is preserved.
    X_train_cpy = np.copy(X_train)
    X_test_cpy = np.copy(X_test)

    # Calculate the mean vector from the train dataset.
    mean = np.mean(X_train_cpy, axis=0)

    # Subtract the mean vector from the train and test datasets.
    X_train_cpy = X_train_cpy - mean
    X_test_cpy = X_test_cpy - mean
    # Now the features have zero mean.

    # Calculate the standard deviation vector from the train dataset.
    std = np.std(X_train_cpy, axis=0)

    # Devide the train and test datasets with the standard deviation vector.
    X_train_cpy = X_train_cpy / std
    X_test_cpy = X_test_cpy / std
    # Now the features have standard deviation of one.

    return X_train_cpy, X_test_cpy


def minmax_normalize(X_train, X_test):
    """ Normalizes the train and test datasets using the min and
    max vectors from the train dataset.

    Parameters
    ----------
    X_train: ndarray
        The train dataset
    X_test: ndarray
        The test dataset

    Returns
    -------
    touple
        Normalized train and test datasets
    """
    # Copy the sets so that the original data is preserved.
    X_train_cpy = np.copy(X_train)
    X_test_cpy = np.copy(X_test)

    # Calculate the min and max vectors from the train dataset.
    mn = np.min(X_train_cpy, axis=0)
    mx = np.max(X_train_cpy, axis=0)

    # Subtract the min vector and devide both the train and test
    # datasets with the difference vector.
    X_train_cpy = (X_train_cpy - mn) / (mx - mn)
    X_test_cpy = (X_test_cpy - mn) / (mx - mn)

    return X_train_cpy, X_test_cpy


class WrongModeException(Exception): pass


def __clean_nan(tx, mode):
    """ Replaces the nan values in tx with the calculated value by mode.

    Parameters
    ----------
    tx: ndarray
        The dataset
    mode: function
        np.mean or np.median

    Returns
    -------
    ndarray
        Cleaned dataset
    """
    # Create a mask for the valid values.
    mask = tx != -999
    # Remove the nan values from tx.
    x = tx * mask
    # Create a matrix that contains only the mean/median values in the
    # entries where there was nan value in the originl matrix.
    y = np.ones(tx.shape) * ~mask * mode(x[np.nonzero(x)], axis=0)
    # Add the mean value entries to the matrix.
    return x + y 


def clean_nan(tx, mode="mean"):
    """ Wrapper around the function that replaces the nan values in tx
    with the calculated value by mode.

    Parameters
    ----------
    tx: ndarray
        The dataset
    mode: string
        "mean" or "median"

    Returns
    -------
    ndarray
        Cleaned dataset
    """
    if mode == "mean":
        return __clean_nan(tx, np.mean)

    if mode =="median":
        return __clean_nan(tx, np.median)

    raise WrongModeException


def map_0_1(y):
    y_copy = y.copy()

    y_copy[y_copy == -1] == 0

    return y_copy


def map_minus_1_1(y):
    y_copy = y.copy()

    y_copy[y_copy == 0] == -1

    return y_copy