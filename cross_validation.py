''' This script does hyperparameter tuning (step size and regularization factor) using cross-validation '''

''' Importing Libraries '''

import numpy as np
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from implementations import least_squares_GD, least_squares_SGD
from implementations import least_squares, ridge_regression
from implementations import logistic_regression, reg_logistic_regression, reg_logistic_regression_L1

from implementations import compute_loss_least_squares, compute_loss_logistic_regression
from implementations import compute_loss_reg_logistic_regression, compute_loss_reg_logistic_regression_L1

from scripts.helpers import predict_labels
from preprocessing import map_0_1


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
    ny = map_0_1(predict_labels(w, tx))
    
    assert ny.shape == y.shape
    assert ny.min() == y.min()
    assert ny.max() == y.max()

    return np.equal(y, ny).astype(int).sum() / y.shape[0]


# given a dataset, model, and all possible params, returns the learned weights w.
def get_model(model, y, tx, initial_w, max_iters, gamma, lambda_, batch_size):
    if model == 'MSE_GD':
        _, w = least_squares_GD(y, tx, initial_w, max_iters, gamma)
        
    elif model == 'MSE_SGD':
        _, w = least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma)
        
    elif model == 'MSE_OPT':
        _, w = least_squares(y, tx)
        
    elif model == 'MSE_OPT_REG':
        _, w = ridge_regression(y, tx, lambda_)
        
    elif model == 'LOG_GD':
        _, w = logistic_regression(y, tx, initial_w, max_iters, gamma)
        
    elif model == 'LOG_REG_GD':
        _, w = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

    elif model == 'LOG_REG_L1':
        _, w = reg_logistic_regression_L1(y, tx, lambda_, initial_w, max_iters, gamma)
    
    else:
        raise UnknownModel
    
    return w


# given all possible params, return the loss of the model
def calculate_loss(model, y, tx, w, lambda_):
    if model == 'MSE_GD':
        return compute_loss_least_squares(y, tx, w)
        
    elif model == 'MSE_SGD':
        return compute_loss_least_squares(y, tx, w)
        
    elif model == 'MSE_OPT':
        return compute_loss_least_squares(y, tx, w)
        
    elif model == 'MSE_OPT_REG':
        return compute_loss_least_squares(y, tx, w)
        
    elif model == 'LOG_GD':
        return compute_loss_logistic_regression(y, tx, w)
        
    elif model == 'LOG_REG_GD':
        return compute_loss_reg_logistic_regression(y, tx, w, lambda_)
        
    elif model == 'LOG_REG_L1':
        return compute_loss_reg_logistic_regression_L1(y, tx, w, lambda_)
        
    else:
        raise UnknownModel


def cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma, batch_size = 1, model = 'LOG_REG_GD'):
    ''' return the loss for regularised logistic regression for the specific kth-fold '''
    ''' splitting the dataset into k-folds for a particular k-value, 
     kth fold - testing set, (k-1 folds) - training set '''
    """
        Executes one fold of CV.
    """
    idx_tr, idx_te = np.append(k_indices[: k].ravel(), k_indices[k + 1:].ravel()), k_indices[k]
    x_train, y_train, x_test, y_test = tx[idx_tr], y[idx_tr], tx[idx_te], y[idx_te]
    
    # get the model weights
    w = get_model(model, y, tx, initial_w, max_iters, gamma, lambda_, batch_size)
    
    # calculate loss
    train_loss = calculate_loss(model, y_train, x_train, w, lambda_)
    test_loss = calculate_loss(model, y_test, x_test, w, lambda_)
    
    # calculate CA
    train_ca, test_ca = accuracy(y_train, x_train, w), accuracy(y_test, x_test, w)
    
    return train_loss, test_loss, train_ca, test_ca


def total_cross_validation(y, tx, k_fold, initial_w, max_iters, gamma, lambda_, seed = 1, batch_size = 1, model = 'LOG_REG_GD'):
    """
        Performs an entire cross validation, and returns LOSS on TRAINING & TEST, and CA on TRAINING & TEST.
    """
    loss_tr, loss_te, ca_tr, ca_te = np.zeros(k_fold), np.zeros(k_fold), np.zeros(k_fold), np.zeros(k_fold)
    
    k_indices = build_k_indices(y, k_fold, seed)
    for k in range(k_fold):
        loss_tr[k], loss_te[k], ca_tr[k], ca_te[k] \
            = cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma, batch_size = batch_size, model = model)
    
    return loss_tr.mean(), loss_te.mean(), ca_tr.mean(), ca_te.mean()  



def gamma_lambda_selection_cv(y, tx, k_fold, initial_w, max_iters, gammas, lambdas, seed = 1, batch_size = 1, metric = 'CA', model = 'LOG_REG_GD'):
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
            print(f"({gamma}, {lambda_})/({len(gammas)}, {len(lambdas)})")
            loss_tr[gamma, lambda_], loss_te[gamma, lambda_], CA_tr[gamma, lambda_], CA_te[gamma, lambda_] \
                = total_cross_validation(y, tx, k_fold, initial_w, max_iters, gammas[gamma], lambdas[lambda_], seed = seed, batch_size = batch_size, model = model)
    
    loss_idx, CA_idx = (-1, -1), (-1, -1)
    for i in range(len(gammas)):
        for j in range(len(lambdas)):
            if loss_idx[0] == -1 or loss_te[i, j] < loss_te[loss_idx[0], loss_idx[1]]:
                loss_idx = (i, j)
            
            if CA_idx[0] == -1 or CA_te[i, j] > CA_te[CA_idx[0], CA_idx[1]]:
                CA_idx = (i, j)
    
    print('CA_te:\n', CA_te)
    print('LOSS_te:\n', loss_te)
    
    # get the CA in CA_tr for training and CA_te for test dataset
    # it is a matrix in [0, len(gammas))x[0, len(lambdas))
    
    
    # return optimal parameters:
    if metric == 'CA':
        # - according to MAX CA
        return gammas[CA_idx[0]], lambdas[CA_idx[1]]
    
    if metric == 'LOSS':
        # - according to MIN LOSS FN
        return gammas[loss_idx[0]], lambdas[loss_idx[1]]
    
#     raise UknownMetricException




def cross_validation_visualization(parameters, loss_tr, loss_te, i, parameter):
    plt.semilogx(parameters, loss_tr, marker=".", color='b', label='train error')
    """visualization the curves of loss_tr and loss_te."""
    
    plt.semilogx(parameters, loss_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("Loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    fig_name = str(parameter) + str(i)
#     plt.savefig(fig_name)    

def plotting_graphs(y, tx, k_fold, initial_w, max_iters, gammas, lambdas, optimal_lambda_, optimal_gamma, i, seed = 1, model = 'LOG_REG_GD'):
       '''plotting graphs between a subset values of lambda, gamma and loss using cross validation'''

    training_losses, testing_losses, training_accuracy, testing_accuracy = [], [], [], []
    
    for gamma in range(len(gammas)):
        loss_tr, loss_te, ca_tr, ca_te = total_cross_validation(y, tx, k_fold, initial_w, max_iters, gammas[gamma], optimal_lambda_, seed = 1, batch_size = 1, model = 'LOG_REG_GD')
        training_losses.append(loss_tr)
        testing_losses.append(loss_te)
        training_accuracy.append(ca_tr)
        testing_accuracy.append(ca_te)
    cross_validation_visualization(gammas, training_losses, testing_losses, i, parameter = 'gamma_') # doing only for losses
    
    
    
    training_losses, testing_losses, training_accuracy, testing_accuracy = [], [], [], []
    for lambda_ in range(len(lambdas)):
        loss_tr, loss_te, ca_tr, ca_te = total_cross_validation(y, tx, k_fold, initial_w, max_iters, lambdas[lambda_], optimal_gamma, seed = 1, batch_size = 1, model = 'LOG_REG_GD')
        training_losses.append(loss_tr)
        testing_losses.append(loss_te)
        training_accuracy.append(ca_tr)
        testing_accuracy.append(ca_te)
    cross_validation_visualization(lambdas, training_losses, testing_losses, i, parameter = 'lambda_')



       