import numpy as np

from helpers import load_csv_data
from helpers import predict_labels, create_csv_submission

from preprocessing import PRI_jet_num_split
from preprocessing import standardize, minmax_normalize
from preprocessing import clean_nan
from preprocessing import map_0_1, map_minus_1_1

from implementations import build_poly

from cross_validation import get_model, calculate_loss, accuracy, total_cross_validation
from cross_validation import gamma_lambda_selection_cv
from cross_validation import plotting_graphs


# Set the seed for reproducibility
np.random.seed(98)


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
print("Shapes:")
print(X_train.shape, y_train.shape, ids_train.shape)
print(X_test.shape, y_test.shape, ids_test.shape)

# Split the datasets into 8 subsets (do not combine groups 2 and 3)
train_subsets = PRI_jet_num_split(y_train, X_train, ids_train, False)
test_subsets = PRI_jet_num_split(y_test, X_test, ids_test, False)

# Print the number of subsets and assert that their sizes are the same
# If not, there is something wrong with the split functionality
print(f"Number of train subsets: { len(train_subsets) }")
print(f"Number of test subsets:  { len(test_subsets) }")
assert len(train_subsets) == len(test_subsets)

# Store the number of subsets for iteration
num_subsets = len(train_subsets)

# Initialize empty arrays for the ids and the predicted labels
# These arrays will be populated by the models
ids = np.array([])
y_pred = np.array([])

# -------------------------------------------------------------
# Optimal parameters
# -------------------------------------------------------------

# Optimal degrees for the polynomial expansion
max_degree = [3, 2, 3, 2, 3, 3, 3, 2]
# Optimal learning rates
gammas_opt = [7e-6, 5e-6, 5e-6, 5e-5, 5e-6, 1e-4, 2e-5, 2e-4]
# Optimal regularization parameters
lambdas_opt = [250, 100, 2, 1, 1e-1, 1e-1, 5, 10]

# -------------------------------------------------------------
# End optimal parameters
# -------------------------------------------------------------

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

    # Build the polynomial features and expand the data
    print(f"Train shape before feature expansion: {str(X_train_subset.shape):>12}   Test shape: {str(X_test_subset.shape):>12}")
    X_train_subset, X_test_subset = build_poly(X_train_subset, max_degree[i]), build_poly(X_test_subset, max_degree[i])
    print(f"Train shape after  feature expansion: {str(X_train_subset.shape):>12}   Test shape: {str(X_test_subset.shape):>12}")
    
    # Set the maximum number of iterations for building the model
    max_iters = 440
    # Set batch size to 1 to enforce SGD
    batch_size = 1
    # Set the initial coefficients randomly
    initial_w = np.random.rand(X_train_subset.shape[1])

    # Get the coefficients of the optimal regularized logistic regression model
    w = get_model("LOG_REG_GD", y_train_subset, X_train_subset, initial_w, max_iters, gammas_opt[i], lambdas_opt[i], 1)

    # Get the predictions
    y_pred_test = np.array(predict_labels(w, X_test_subset))

    # Insert the ids and predictions to the ids and y_pred arrays
    ids = np.concatenate((ids, ids_test_subset))
    y_pred = np.concatenate((y_pred, y_pred_test))

# Sort the ids and y_pred arrays
ids, y_pred = sort_arr(ids, y_pred)

# Create the submission CSV file
create_csv_submission(ids, y_pred, sumbission_fname)
