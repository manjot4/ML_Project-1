import numpy as np
from scripts.helpers import load_csv_data

np.random.seed(1)

train_fname = "data/train.csv"
test_fname = "data/test.csv"

y_train, X_train, ids_train = load_csv_data(train_fname)
y_test, X_test, ids_test = load_csv_data(test_fname)

print(X_train.shape, y_train.shape, ids_train.shape)
print(X_test.shape, y_test.shape, ids_test.shape)

from preprocessing import PRI_jet_num_split
from preprocessing import standardize, minmax_normalize
from preprocessing import clean_nan
from implementations import least_squares_SGD
from scripts.helpers import predict_labels

combine_vals = False

train_subsets = PRI_jet_num_split(y_train, X_train, ids_train, combine_vals)
test_subsets = PRI_jet_num_split(y_test, X_test, ids_test, combine_vals)

print(f"Number of train subsets: { len(train_subsets) }")
print(f"Number of test subsets:  { len(test_subsets) }")
print()

assert len(train_subsets) == len(test_subsets)

num_subsets = len(train_subsets)

ids = []
y_pred = []

for i in range(num_subsets):
    y_train_subset, X_train_subset, ids_train_subset = train_subsets[i]
    y_test_subset, X_test_subset, ids_test_subset = test_subsets[i]
    
    X_train_subset, X_test_subset = standardize(X_train_subset, X_test_subset)

    N, D = X_train_subset.shape

    initial_w = np.random.randn(D)
    gamma = 0.003
    
    print(f"Train shape: {str(X_train_subset.shape):>12}   Test shape: {str(X_test_subset.shape):>12}")
    
    loss, w = least_squares_SGD(y_train_subset, X_train_subset, initial_w, 1, N, gamma)
    labels = predict_labels(w, X_test_subset)

    print(f"Number of samples:                      { len(labels) }")
    print(f"Number of correctly classified samples: { np.sum(labels - y_test_subset == 0) }")
    
    ids.extend(ids_test_subset)
    y_pred.extend(labels)
