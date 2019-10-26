"""
This code is used to find the optimal values for
the hyperparameters of the models with grid search.
Models are evaluated with 5-fold cross-validation.
"""

import numpy as np

from preprocessing import PRI_jet_num_split
from preprocessing import standardize, minmax_normalize
from preprocessing import clean_nan
from preprocessing import map_0_1, map_minus_1_1

from cross_validation import get_model, calculate_loss, accuracy, total_cross_validation

from implementations import build_poly

from helpers import load_csv_data
from helpers import predict_labels, create_csv_submission

from cross_validation import gamma_lambda_selection_cv
from cross_validation import plotting_graphs


def sort_arr(ids, y_pred):
    """ Utility function for sorting predictions by the id.

    Parameters
    ----------
    ids: ndarray
        The ids
    y_pred: ndarray
        The predictions

    Returns
    -------
    Tuple (ndarray, ndarray)
        Sorted labels and predictions
    """
    idx = ids.argsort()
    return ids[idx], y_pred[idx]


# Locations of the train/test data and the submission files
train_fname = "data/train.csv"
test_fname = "data/test.csv"
sumbission_fname = "data/submission.csv"


# Load the train/test data
y_train, X_train, ids_train = load_csv_data(train_fname)
y_test, X_test, ids_test = load_csv_data(test_fname)


# Print out the shapes for convinience
print("Shapes")
print(X_train.shape, y_train.shape, ids_train.shape)
print(X_test.shape, y_test.shape, ids_test.shape)


# Split the datasets into 8 subsets
combine_vals = False
train_subsets = PRI_jet_num_split(y_train, X_train, ids_train, combine_vals)
test_subsets = PRI_jet_num_split(y_test, X_test, ids_test, combine_vals)

# Print the number of subsets and assert that their sizes are the same
# If not, there is something wrong with the split functionality
print(f"Number of train subsets: { len(train_subsets) }")
print(f"Number of test subsets:  { len(test_subsets) }")

# Store the number of subsets for iteration
assert len(train_subsets) == len(test_subsets)
num_subsets = len(train_subsets)

# Initialize empty arrays for the ids and the predicted labels
# These arrays will be populated by the models
ids = np.array([])
y_pred = np.array([])


# -------------------------------------------------------------
# Tweak the possible hyperparameters
# -------------------------------------------------------------

# Expetected testing / training measure (accuracy or loss)
exp_measure_tr, exp_measure_te = 0, 0

# Degrees for the polynomial expansion
max_degree = [3, 2, 3, 2, 3, 3, 3, 2]

# Perentage of features that are left after feature selection
fs_perc = [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6] # [1.0, 0.9, 0.9, 0.9, 0.7, 0.8, 0.8, 0.65]

# Possible step size and regularization coefficients
gammas, lambdas = \
[[3e-6], [5e-6], [1e-6, 5e-6, 1e-5], [5e-5], [5e-6], [1e-4], [2e-5], [2e-4]],\
[[150], [100], [0, 1, 10], [1], [1e-1], [1e-1], [5], [10]]

# Set seed for reproducibility
np.random.seed(98)


# Iterate over each subset and build a model
# The predictions of every single model are combined
for i in range(num_subsets):
    # Extract the train/test subsets
    y_train_subset, X_train_subset, ids_train_subset = train_subsets[i]
    y_test_subset, X_test_subset, ids_test_subset = test_subsets[i]
    
    # Map the categorical output labels into [0, 1]
    y_train_subset = map_0_1(y_train_subset)
    # Standardize the data
    X_train_subset, X_test_subset = standardize(X_train_subset, X_test_subset)
    print(f"Train shape before feature expansion: {str(X_train_subset.shape):>12}   Test shape: {str(X_test_subset.shape):>12}")
    # Build the polynomial features and expand the data
    X_train_subset, X_test_subset = build_poly(X_train_subset, max_degree[i]), build_poly(X_test_subset, max_degree[i])
    print(f"Train shape after  feature expansion: {str(X_train_subset.shape):>12}   Test shape: {str(X_test_subset.shape):>12}")
    
    # Set n_best_features to X_train_subset.shape[1] if you don't want feature selection
    n_best_features = round(fs_perc[i] * X_train_subset.shape[1])
    D = n_best_features
    N, _ = X_train_subset.shape
    
    # Accuracy by predicting the majority class in the training dataset
    CA_one = y_train_subset.sum() / N
    CA_zero = 1 - CA_one
    CA_baseline = max(CA_zero, CA_one)
    
    # Parameters for model with L1 regularization
    max_iters_fs = 300
    gamma_fs, lambda_fs = 1e-7, 1e2
    model_fs = 'LOG_REG_L1'
    
    # Build the model
    initial_w_fs = np.random.randn(X_train_subset.shape[1])
    w_fs = get_model(model_fs, y_train_subset, X_train_subset, initial_w_fs, max_iters_fs, gamma_fs, lambda_fs, 1)
    
    # Choose the best features
    features = np.argsort(abs(w_fs))[::-1][:n_best_features]
    print(w_fs.min(), w_fs.max(), w_fs.mean())
    
    # Select only the best features
    X_train_subset, X_test_subset = X_train_subset[:, features], X_test_subset[:, features]
    
    # Parameters for the main model
    k_fold = 5
    max_iters = 500
    seed, batch_size = 17, 1
    metric, model = 'CA', 'LOG_REG_GD'
    
    # Build the model
    initial_w = np.random.randn(D)
    optimal_gamma, optimal_lambda_, measure_tr, measure_te = \
        gamma_lambda_selection_cv(y_train_subset, X_train_subset, k_fold, initial_w, max_iters, gammas[i], lambdas[i],
                                  seed = seed, batch_size = batch_size, metric = metric, model = model)
    print('CA_bs:', CA_baseline)
    print('Iter:', i, ' Best gamma:', optimal_gamma, ' Best lambda:', optimal_lambda_, '\n')
    
    # Update the expected training error
    exp_measure_tr += measure_tr * X_train_subset.shape[0] / X_train.shape[0]
    exp_measure_te += measure_te * X_test_subset.shape[0] / X_test.shape[0]
    
    # Build the model with the best hyperparameters
    w = get_model(model, y_train_subset, X_train_subset, initial_w, max_iters, optimal_gamma, optimal_lambda_, batch_size)
    
    # Get predictions
    y_pred_test = np.array(map_minus_1_1(predict_labels(w, X_test_subset)))
    
    # Insert the ids and predictions to the ids and y_pred arrays
    ids = np.concatenate((ids, ids_test_subset))
    y_pred = np.concatenate((y_pred, y_pred_test))

# Sort the ids and y_pred arrays
ids, y_pred = sort_arr(ids, y_pred)
# Create the submission CSV file
create_csv_submission(ids, y_pred, sumbission_fname)

print("Expected training accuracy / loss:", exp_measure_tr)
print("Expected test accuracy / loss:", exp_measure_te)