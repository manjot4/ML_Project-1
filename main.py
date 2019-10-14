# need to wrap main in a function such it runs all possible combinations 
# need to store optimal gamma and lambda for different possible combinations

import numpy as np
from preprocessing import PRI_jet_num_split
from preprocessing import standardize, minmax_normalize
from preprocessing import clean_nan
from preprocessing import map_0_1, map_minus_1_1
from implementations import reg_logistic_regression, logistic_regression
from scripts.helpers import load_csv_data
from scripts.helpers import predict_labels, create_csv_submission
from cross_validation import gamma_lambda_selection_cv, optimal_gamma_selection, optimal_lambda_selection, verifying_accuracy_cv

np.random.seed(1)

gammas = [1e-6, 1e-6, 1e-6, 1e-5, 1e-6, 1e-05, 1e-05, 1e-05]
lambdas_ = [1000.0, 0.001, 100.0, 0.001, 100.0, 100.0, 0.001, 100.0]

train_fname = "data/train.csv"
test_fname = "data/test.csv"
sumbission_fname = "data/submission.csv"

y_train, X_train, ids_train = load_csv_data(train_fname)
y_test, X_test, ids_test = load_csv_data(test_fname)

print("Shapes")
print(X_train.shape, y_train.shape, ids_train.shape)
print(X_test.shape, y_test.shape, ids_test.shape)
print()

combine_vals = False

train_subsets = PRI_jet_num_split(y_train, X_train, ids_train, combine_vals)
test_subsets = PRI_jet_num_split(y_test, X_test, ids_test, combine_vals)

print(f"Number of train subsets: { len(train_subsets) }")
print(f"Number of test subsets:  { len(test_subsets) }")
print()

assert len(train_subsets) == len(test_subsets)

num_subsets = len(train_subsets)

ids = np.array([])
y_pred = np.array([])

def sort_arr(ids, y_pred):
    idx = ids.argsort()
    return ids[idx], y_pred[idx]


# Cross Validation 
lambdas = np.logspace(-4, 0, 30) # if needed, change the space as appropriate
gammas = np.logspace(-3, 3, 30) # if needed, change the space as appropriate

training_accuracy = []
testing_accuracy = []

for i in range(num_subsets):
    y_train_subset, X_train_subset, ids_train_subset = train_subsets[i]
    y_test_subset, X_test_subset, ids_test_subset = test_subsets[i]

    y_train_subset = map_0_1(y_train_subset)

    X_train_subset, X_test_subset = standardize(X_train_subset, X_test_subset)

    N, D = X_train_subset.shape

    initial_w = np.random.randn(D)
    gamma = gammas[i]
    lambda_ = lambdas_[i]

    # need to chose optimal lambda and optimal gamma together
    k_fold = 4 # can experiment with different numbers
    max_iters = 500
    optimal_lambda_, optimal_gamma = gamma_lambda_selection_cv(y_train_subset, X_train_subset, k_fold, lambdas, initial_w, max_iters, gammas)

    # selecting either gamma or lambda using cross validation
    # model = 'logistic_regression' #change as appropriate
    optimal_gamma = optimal_gamma_selection(y_train_subset, X_train_subset, k_fold, initial_w, max_iters, gammas, model)
    optimal_lambda_ = optimal_lambda_selection(y_train_subset, X_train_subset, k_fold, initial_w, max_iters, lambdas, model)

    print(f"Train shape: {str(X_train_subset.shape):>12}   Test shape: {str(X_test_subset.shape):>12}")
    print()
        
    loss, w = reg_logistic_regression(y_train_subset, X_train_subset, lambda_, initial_w, 2000, gamma)

    # accuracy using cross_validation for each subset
    train_acc, test_acc = verifying_accuracy_cv(y_train, X_train, k_fold, w)
    training_accuracy.append(train_acc)
    testing_accuracy.append(test_acc)

    y_pred_test = predict_labels(w, X_test_subset)

    ids = np.concatenate((ids, ids_test_subset))
    y_pred = np.concatenate((y_pred, y_pred_test))


training_accuracy = sum(training_accuracy)/float(num_subsets)
testing_accuracy = sum(testing_accuracy)/float(num_subsets)

ids, y_pred = sort_arr(ids, y_pred)

create_csv_submission(ids, y_pred, sumbission_fname)