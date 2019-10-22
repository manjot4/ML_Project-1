""" This file contains implementations for hyperparameter tuning (step size and regularization factor) using cross-validation """

import numpy as np
import matplotlib.pyplot as plt
from implementations import least_squares_GD, least_squares_SGD
from implementations import least_squares, ridge_regression
from implementations import logistic_regression, reg_logistic_regression, reg_logistic_regression_L1

from implementations import compute_loss_least_squares, compute_loss_logistic_regression
from implementations import compute_loss_reg_logistic_regression, compute_loss_reg_logistic_regression_L1

from helpers import predict_labels
from preprocessing import map_0_1


class UnknownModel(Exception): pass
class UknownMetricException(Exception): pass


def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold.
    
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def accuracy(y, tx, w):
    """ Computes the accuracy of a model.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The learned weights

    Returns
    -------
    float
        The accuracy  
    """
    ny = map_0_1(predict_labels(w, tx))
    
    assert ny.shape == y.shape
    assert y.min() in [0, 1]
    assert y.max() in [0, 1]
    assert ny.min() in [0, 1]
    assert ny.max() in [0, 1]

    return np.equal(y, ny).astype(int).sum() / y.shape[0]


def get_model(model, y, tx, initial_w, max_iters, gamma, lambda_, batch_size):
    """ Returns the learned weights 'w' (last weight vector) and
    the corresponding loss function by a given model.

    Parameters
    ----------
    model: string
        The model
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    initial_w: ndarray
        The initial weights
    max_iters: integer
        The number of steps to run
    gamma: integer
        The step size
    lambda_: integer
        The regularization parameter
    batch_size: integer
        The batch size

    Returns
    -------
    tuple
        The learned weights
    """
    if model == "MSE_GD":
        _, w = least_squares_GD(y, tx, initial_w, max_iters, gamma)
        
    elif model == "MSE_SGD":
        _, w = least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma)
        
    elif model == "MSE_OPT":
        _, w = least_squares(y, tx)
        
    elif model == "MSE_OPT_REG":
        _, w = ridge_regression(y, tx, lambda_)
        
    elif model == "LOG_GD":
        _, w = logistic_regression(y, tx, initial_w, max_iters, gamma)
        
    elif model == "LOG_REG_GD":
        _, w = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

    elif model == "LOG_REG_L1":
        _, w = reg_logistic_regression_L1(y, tx, lambda_, initial_w, max_iters, gamma)
    
    else:
        raise UnknownModel
    
    return w


def calculate_loss(model, y, tx, w, lambda_):
    """ Wrapper arround the loss calculation functions.
    
    Parameters
    ----------
    model: string
        The model
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The weights
    lambda_: integer
        The regularization parameter

    Returns
    -------
    tuple
        The loss
    """
    if model == "MSE_GD":
        return compute_loss_least_squares(y, tx, w)
        
    elif model == "MSE_SGD":
        return compute_loss_least_squares(y, tx, w)
        
    elif model == "MSE_OPT":
        return compute_loss_least_squares(y, tx, w)
        
    elif model == "MSE_OPT_REG":
        return compute_loss_least_squares(y, tx, w)
        
    elif model == "LOG_GD":
        return compute_loss_logistic_regression(y, tx, w)
        
    elif model == "LOG_REG_GD":
        return compute_loss_reg_logistic_regression(y, tx, w, lambda_)
        
    elif model == "LOG_REG_L1":
        return compute_loss_reg_logistic_regression_L1(y, tx, w, lambda_)
        
    else:
        raise UnknownModel


def cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma, batch_size=1, model="LOG_REG_GD"):
    """ Returns the loss/accuracy for a given model for the k-th fold.
    
    Executes one fold of CV.
    
    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    k_indices: ndarray
        The indices for k-fold
    k: integer
        The fold index
    lambda_: integer
        The regularization parameter
    initial_w: ndarray
        The learned weights
    max_iters: integer
        The number of steps to run
    gamma: integer
        The step size
    batch_size: integer
        The batch size
    model: string
        The model
    
    Returns
    -------
    tuple
        The loss and accuracy
    """
    idx_tr, idx_te = np.append(k_indices[: k].ravel(), k_indices[k + 1:].ravel()), k_indices[k]
    x_train, y_train, x_test, y_test = tx[idx_tr], y[idx_tr], tx[idx_te], y[idx_te]
    
    # Get the model weights
    w = get_model(model, y_train, x_train, initial_w, max_iters, gamma, lambda_, batch_size)
    
    # Calculate loss
    train_loss = calculate_loss(model, y_train, x_train, w, lambda_)
    test_loss = calculate_loss(model, y_test, x_test, w, lambda_)
    
    # Calculate CA
    train_ca, test_ca = accuracy(y_train, x_train, w), accuracy(y_test, x_test, w)
    
    return train_loss, test_loss, train_ca, test_ca


def total_cross_validation(y, tx, k_fold, initial_w, max_iters, gamma, lambda_, seed=1, batch_size=1, model="LOG_REG_GD"):
    """ Performs an entire cross validation.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    k_fold: integer
        The fold index
    initial_w: ndarray
        The learned weights
    max_iters: integer
        The number of steps to run
    gamma: integer
        The step size
    lambda_: integer
        The regularization parameter
    seed: integer
        The seed
    batch_size: integer
        The batch size
    model: string
        The model    

    Returns
    -------
    tuple
        Mean loss and mean accuracy on training and test set
    
    """
    loss_tr, loss_te, ca_tr, ca_te = np.zeros(k_fold), np.zeros(k_fold), np.zeros(k_fold), np.zeros(k_fold)
    
    k_indices = build_k_indices(y, k_fold, seed)
    for k in range(k_fold):
        loss_tr[k], loss_te[k], ca_tr[k], ca_te[k] \
            = cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma, batch_size = batch_size, model = model)
    
    return loss_tr.mean(), loss_te.mean(), ca_tr.mean(), ca_te.mean()  


def gamma_lambda_selection_cv(y, tx, k_fold, initial_w, max_iters, gammas, lambdas, seed=1, batch_size=1, metric="CA", model="LOG_REG_GD"):
    """ Implements cross validation to fine tune gamma and lambda.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    k_fold: integer
        The fold index
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gammas: ndarray
        The set of gammas from which optimal gamma is chosen
    lambdas: ndarray
        The set of lambdas from which optimal lambda is chosen
    seed: integer
        The seed
    batch_size: integer
        The batch size
    metric: string
        The metric
    model: string
        The model 

    Returns
    -------
    tuple
        Optimal lambda, optimal gamma, CA or loss in traning set, CA or loss in test set
    """

    # 1. Split data in k-folds
    # 2. Store minimum loss values for a particular lambda and gamma
    # 3. Chose optimal lambda and gamma value with minimum loss

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
    
    print("CA_tr:\n", CA_tr)
    print("CA_te:\n", CA_te)
    print("LOSS_tr:\n", loss_tr)
    print("LOSS_te:\n", loss_te)
    
    print("CA_tr:", CA_tr[CA_idx])
    print("CA_te:", CA_te[CA_idx])
    
    # Get the CA in CA_tr for training and CA_te for test dataset
    # It is a matrix in [0, len(gammas))x[0, len(lambdas))
    
    # Return optimal parameters
    if metric == "CA":
        # According to MAX CA
        return gammas[CA_idx[0]], lambdas[CA_idx[1]], CA_tr[CA_idx], CA_te[CA_idx]
    
    if metric == "LOSS":
        # According to MIN LOSS FN
        return gammas[loss_idx[0]], lambdas[loss_idx[1]], loss_tr, loss_te
    
    raise UknownMetricException


def cross_validation_visualization(parameters, loss_tr, loss_te, i, parameter):
    """ Visualizes the curves of loss_te.
    
    """
    plt.semilogx(parameters, loss_tr, marker=".", color="b", label="train error")
    plt.semilogx(parameters, loss_te, marker=".", color="r", label="test error")
    plt.xlabel("Lambda")
    plt.ylabel("Loss")
    plt.title("Cross-validation")
    plt.legend(loc=2)
    plt.grid(True)
    # fig_name = str(parameter) + str(i)
    # plt.savefig(fig_name)    


def plotting_graphs(y, tx, k_fold, initial_w, max_iters, gammas, lambdas, optimal_lambda_, optimal_gamma, i, seed=1, model="LOG_REG_GD"):
    """ Plots graphs between a subset values of lambda, gamma, and loss using cross validation.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The input matrix
    k_fold: integer
        The fold index
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gammas: ndarray
        The set of gammas from which optimal gamma is chosen
    lambdas: ndarray
        The set of lambdas from which optimal lambda is chosen
    optimal_lambda_: integer
        The optimal lambda
    optimal_gamma: integer
        The optimal gamma
    seed: integer
        The seed
    model: string
        The model       
    """
    training_losses, testing_losses, training_accuracy, testing_accuracy = [], [], [], []
    
    for gamma in range(len(gammas)):
        loss_tr, loss_te, ca_tr, ca_te = total_cross_validation(y, tx, k_fold, initial_w, max_iters, gammas[gamma], optimal_lambda_, seed=1, batch_size=1, model="LOG_REG_GD")
        training_losses.append(loss_tr)
        testing_losses.append(loss_te)
        training_accuracy.append(ca_tr)
        testing_accuracy.append(ca_te)
    cross_validation_visualization(gammas, training_losses, testing_losses, i, parameter = "gamma_") # doing only for losses

    training_losses, testing_losses, training_accuracy, testing_accuracy = [], [], [], []
    for lambda_ in range(len(lambdas)):
        loss_tr, loss_te, ca_tr, ca_te = total_cross_validation(y, tx, k_fold, initial_w, max_iters, lambdas[lambda_], optimal_gamma, seed=1, batch_size=1, model="LOG_REG_GD")
        training_losses.append(loss_tr)
        testing_losses.append(loss_te)
        training_accuracy.append(ca_tr)
        testing_accuracy.append(ca_te)
    cross_validation_visualization(lambdas, training_losses, testing_losses, i, parameter = "lambda_")
 