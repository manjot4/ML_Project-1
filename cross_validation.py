''' This script does hyperparameter tuning (step size and regularization factor) using cross-validation '''

''' Importing Libraries '''

import numpy as np
from implementations import reg_logistic_regression, compute_loss_reg_logistic_regression
from implementations import least_squares_GD, least_squares_SGD, logistic_regression
from implementations import compute_loss_least_squares, compute_loss_logistic_regression
from implementations import ridge_regression
from helpers import predict_labels


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def accuracy(y, tx, w):
    ny = tx.dot(w)
    assert ny.shape == y.shape
    return np.equal(y, ny).astype(int).sum() / y.shape[0]


def cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma, model = 'LOG_REG_GD'):
    ''' return the loss for regularised logistic regression for the specific kth-fold '''
    ''' splitting the dataset into k-folds for a particular k-value, 
     kth fold - testing set, (k-1 folds) - training set '''
    """
        Executes one fold of CV.
    """
    idx_tr, idx_te = np.append(k_indices[: k].ravel(), k_indices[k + 1:].ravel()), k_indices[k]
    x_train, y_train, x_test, y_test = tx[idx_tr], y[idx_tr], tx[idx_te], y[idx_te]
    
    if model == 'MSE_GD':
        raise NeedToImplement
    
    if model == 'MSE_SGD':
        raise NeedToImplement
    
    if model == 'MSE_OPT':
        raise NeedToImplement
    
    if model == 'MSE_OPT_REG':
        raise NeedToImplement
    
    if model == 'LOG_GD':
        raise NeedToImplement
    
    if model == 'LOG_REG_GD':
        # Regularised Logistic Regression, getting weights
        _, weight = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)

        # calculating loss for train and test data     
        train_loss = compute_loss_reg_logistic_regression(y_train, x_train, lambda_, weight)
        test_loss = compute_loss_reg_logistic_regression(y_test, x_test, lambda_, weight)

        train_ca = accuracy(y_train, x_train, weight)
        test_ca = accuracy(y_test, x_test, weight)
        
        return train_loss, test_loss, train_ca, test_ca
    
    if model == 'LOG_REG_SGD':
        raise NeedToImpelement
    
    raise UknownModelError


def total_cross_validation(y, tx, k_fold, initial_w, max_iters, gamma, lambda_, seed = 1):
    """
        Performs an entire cross validation, and returns LOSS on TRAINING & TEST, and CA on TRAINING & TEST.
    """
    loss_tr, loss_te, ca_tr, ca_te = np.zeros(k_fold), np.zeros(k_fold), np.zeros(k_fold), np.zeros(k_fold)
    
    k_indices = build_k_indices(y, k_fold, seed)
    for k in range(k_fold):
        loss_tr[k], loss_te[k], ca_tr[k], ca_te[k] \
            = cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma)
    
    return loss_tr.mean(), loss_te.mean(), ca_tr.mean(), ca_te.mean()


def gamma_lambda_selection_cv(y, tx, k_fold, initial_w, max_iters, gammas, lambdas, seed = 1, metric = 'CA'):
    """ Implementing cross_validation to hypertune gamma and lambda 

    Parameters
    ----------
    y: ndarray
        Labels
    tx: ndarray
        Input Matrix
    lambdas: Regularisation Parameter
        set of lambdas from which optimal lambda_ is chosen	
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gamma: The step size
        set of gammas from which optimal gamma is chosen
    Returns
    -------
    tuple
        optimal_lambda_ and optimal_gamma
    """
    
    # split data in k fold
    # list for storing minimum loss values for a particular lambda and gamma
    # and then choosing optimal lambda and gamma value with minimum loss
    
    loss_tr, loss_te = np.zeros((len(gammas), len(lambdas))), np.zeros((len(gammas), len(lambdas)))
    CA_tr, CA_te = np.zeros((len(gammas), len(lambdas))), np.zeros((len(gammas), len(lambdas)))
    for gamma in range(len(gammas)):
        for lambda_ in range(len(lambdas)):
            loss_tr[gamma, lambda_], loss_te[gamma, lambda_], CA_tr[gamma, lambda_], CA_te[gamma, lambda_] \
                = total_cross_validation(y, tx, k_fold, initial_w, max_iters, gammas[gamma], lambdas[lambda_], seed = seed)
    
    loss_idx, CA_idx = (-1, -1), (-1, -1)
    for i in range(len(gammas)):
        for j in range(len(lambdas)):
            if loss_idx[0] == -1 or loss_te[i, j] < loss_te[loss_idx[0], loss_idx[1]]:
                loss_idx = (i, j)
            
            if CA_idx[0] == -1 or CA_te[i, j] > CA_te[CA_idx[0], CA_idx[1]]:
                CA_idx = (i, j)
    
    # get the CA in CA_tr for training and CA_te for test dataset
    # it is a matrix in [0, len(gammas))x[0, len(lambdas))
    
    
    # return optimal parameters:
    if metric == 'CA':
        # - according to MAX CA
        return gammas[CA_idx[0]], lambdas[CA_idx[1]]
    
    if metric == 'LOSS':
        # - according to MIN LOSS FN
        return gammas[loss_idx[0]], lambdas[loss_idx[1]]
    
    raise UknownMetricException

# This is for graphs, will use the code from exercises - will generate graphs later this week. 
# cross_validation_visualization(lambdas, log_reg_tr, log_reg_te) - plots from ex4.