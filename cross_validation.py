# Cross - Validation
# In progress, need to comment in the different format and also chose optimal gamma
# this script for now only uses Regularised Logistic Regression

import numpy as np
from implementations import reg_logistic_regression, compute_loss_reg_logistic_regression


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma):
    ''' return the loss for logistic regression '''
    k_test = k_indices[k]
    k_train = k_indices[(np.arange(k_indices.shape[0]) != k)]
    k_train = k_train.reshape(-1)
    x_train = tx[k_train]
    x_test = tx[k_test]
    y_train = y[k_train]
    y_test = y[k_test]

    # Logistic Regression, getting weights
    _, weight = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    
    # calculating loss for train and test data     
    loss_tr = compute_loss_reg_logistic_regression(y_train, x_train, lambda_, weight)
    loss_te = compute_loss_reg_logistic_regression(y_test, x_test, lambda_, weight)
    return loss_tr, loss_te


# choosing best lambda 
def best_lambda_selection(y, tx, k_fold, lambdas, initial_w, max_iters, gamma):
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    log_reg_tr = [] 
    log_reg_te = []
    for j in range(len(lambdas)):
        tr_loss = []
        te_loss = []
        for j in range(k_fold):
            tr, te = cross_validation(y, tx, k_indices, j, lambdas[j], initial_w, max_iters, gamma)
            tr_loss.append(tr)
            te_loss.append(te)
        log_reg_tr.append(np.mean(tr_loss))
        log_reg_te.append(np.mean(te_loss))
    optimal_lambda_ = lambdas[np.argmin(log_reg_te)]     
    return optimal_lambda_


# def best_alpha_selection():    

# cross_validation_visualization(lambdas, log_reg_tr, log_reg_te) - plots from ex4. 