""" This file contains implementations (loss and gradient computation, and polynomial expansion) of all the models used in the project """

import numpy as np


# Main functions

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Implementation of linear regression using gradient descent.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gamma:
        The step size

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    w = initial_w
    losses = []

    for _ in range(max_iters):
        gradient = compute_gradient_least_squares(y, tx, w)
        loss = compute_loss_least_squares(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)

    return w, losses[-1]


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """ Implementation of linear regression using stochastic gradient descent.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    initial_w: ndarray
        The initial weight vector
    batch_size: integer
        The batch size
    max_iters: integer
        The number of steps to run
    gamma:
        The step size

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    w = initial_w
    losses = []
    
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size = batch_size, num_batches = 1):
            gradient = compute_gradient_least_squares(minibatch_y, minibatch_tx, w)
            loss = compute_loss_least_squares(y, tx, w)
            w = w - gamma * gradient
            losses.append(loss)

    return w, losses[-1]


def least_squares(y, tx):
    """ Implementation of the closed form solution for least squares.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w, 1 / y.shape[0] * np.sum(np.power(y - tx.dot(w), 2))


def ridge_regression(y, tx, lambda_):
    """ Implementation of the closed form solution for ridge regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.identity(tx.shape[1]), tx.T.dot(y))
    return w, 1 / y.shape[0] * np.sum(np.power(y - tx.dot(w), 2))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Implementation of logistic regression using gradient descent.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gamma:
        The step size

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    w = initial_w
    losses = []

    for _ in range(max_iters):
        gradient = compute_gradient_logistic_regression(y, tx, w)
        loss = compute_loss_logistic_regression(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)

    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Implementation of regularized logistic regression using gradient descent.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    lambda_: integer
        The regularization parameter
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gamma:
        The step size

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    
    w = initial_w
    losses = []

    for _ in range(max_iters):
        gradient = compute_gradient_reg_logistic_regression(y, tx, w, lambda_)
        loss = compute_loss_reg_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
        losses.append(loss)

    return w, losses[-1]


def reg_logistic_regression_L1(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Implementation of regularized logistic regression using gradient descent.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    lambda_: integer
        The regularization parameter
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gamma:
        The step size

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    w = initial_w
    losses = []

    for _ in range(max_iters):
        gradient = compute_gradient_reg_logistic_regression_L1(y, tx, w, lambda_)
        loss = compute_loss_reg_logistic_regression_L1(y, tx, w, lambda_)
        w = w - gamma * gradient
        losses.append(loss)

    return w, losses[-1]



def least_squares_GD_L1(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Implementation of linear regression using gradient descent.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    lambda_: integer
        The regularization parameter
    initial_w: ndarray
        The initial weight vector
    max_iters: integer
        The number of steps to run
    gamma:
        The step size

    Returns
    -------
    tuple
        The last loss and learned weights
    """
    w = initial_w
    losses = []

    for _ in range(max_iters):
        gradient = compute_gradient_least_squares_L1(y, tx, w, lambda_)
        loss = compute_loss_least_squares_L1(y, tx, w, lambda_)
        w = w - gamma * gradient
        losses.append(loss)

    return w, losses[-1]



# Auxiliary functions

def compute_loss_least_squares(y, tx, w):
    """ Computes the loss for linear regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The weight vector

    Returns
    -------
    tuple
        Loss
    """
    N = tx.shape[0]

    e = y - np.matmul(tx, w)

    return 1.0 / (2 * N) * np.sum(e ** 2)


def compute_loss_logistic_regression(y, tx, w):
    """ Computes the loss for logistic regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The weight vector

    Returns
    -------
    tuple
        Loss
    """
    eps = 1e-5
    predictions = sigmoid(tx.dot(w))
    assert predictions.min() > -eps # make sure numbers are close to 0
    loss = y.T.dot(np.log(predictions + eps)) + (1 - y).T.dot(np.log(1 - predictions + eps)).squeeze()
    return -loss



def compute_loss_reg_logistic_regression(y, tx, w, lambda_):
    """ Computes the loss for regularized logistic regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    lambda_: integer
        The regularization parameter
    w: ndarray
        The weight vector

    Returns
    -------
    tuple
        Loss
    """
    return compute_loss_logistic_regression(y, tx, w) + (lambda_ / 2) * np.matmul(w.T, w).squeeze()


def compute_loss_reg_logistic_regression_L1(y, tx, w, lambda_):
    """ Computes the loss for regularized logistic regression with L1 norm.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    lambda_: integer
        The regularization parameter
    w: ndarray
        The weight vector

    Returns
    -------
    tuple
        Loss
    """
    return compute_loss_logistic_regression(y, tx, w) + lambda_ * np.sum(np.absolute(w))


def compute_loss_least_squares_L1(y, tx, w, lambda_):
    """ Computes the loss for linear regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The weight vector
    lamda_: integer
        The regularization parameter

    Returns
    -------
    tuple
        Loss
    """
    return compute_loss_least_squares(y, tx, w) + lambda_ * np.sum(np.absolute(w))



def compute_gradient_least_squares(y, tx, w):
    """ Computes the gradient for linear regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The weight vector

    Returnsty
    -------
    ndarray
        The gradient
    """
    N = tx.shape[0]

    e = y - np.matmul(tx, w)

    return - 1.0 / N * np.matmul(tx.T, e)


def compute_gradient_logistic_regression(y, tx, w):
    """ Computes the gradient for logistic regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The weight vector

    Returns
    -------
    ndarray
        The gradient
    """
    e = sigmoid(np.matmul(tx, w)) - y
    return np.matmul(tx.T, e)


def compute_gradient_reg_logistic_regression(y, tx, w, lambda_):
    """ Computes the gradient for regularized logistic regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    lambda_: integer
        The regularization parameter
    w: ndarray
        The weight vector

    Returns
    -------
    ndarray
        The gradient
    """
    return compute_gradient_logistic_regression(y, tx, w) + lambda_ * w


def compute_gradient_reg_logistic_regression_L1(y, tx, w, lambda_):
    """ Computes the gradient for regularized logistic regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    lambda_: integer
        The regularization parameter
    w: ndarray
        The weight vector

    Returns
    -------
    ndarray
        The gradient
    """
    return compute_gradient_logistic_regression(y, tx, w) + lambda_ * np.sign(w)


def compute_gradient_least_squares_L1(y, tx, w, lambda_):
    """ Computes the gradient for linear regression.

    Parameters
    ----------
    y: ndarray
        The labels
    tx: ndarray
        The feature matrix
    w: ndarray
        The weight vector
    lambda_: integer
        The regularization parameter    

    Returns
    -------
    ndarray
        The gradient
    """
    return compute_gradient_least_squares(y, tx, w) + lambda_ * np.sign(w)


def sigmoid(z):
    """ Computes the sigmoid function.

    Parameters
    ----------
    z: vector
        Input value

    Returns
    -------
    integer
        Sigmoid of input
    """
    return 1.0 / (1 + np.exp(-z))


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, d):
    """ Polynomial basis function for input data x and degree from 0 to 'd'.
    
    Takes a dataset of N examples and D features (NxD array) and creates
    polynomials of the features up to degree 'd'.
    
    The first column is always 1 as it represents all polynomials with
    degree 0 for all D features. Then the first 'd' columns represent
    the polynomials of the first feature, the following 'd' columns 
    represent the polynomials for the second feature and so on.
    
    Thus, the returned dataset has size Nx(1 + D * d).

    Parameters
    ----------
    x: ndarray
        The dataset
    d: integer
        The degree

    Returns
    -------
    ndarray
        Expanded dataset
    """
    
    # Handle the case when D is equal to 1
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    ret = np.ones((x.shape[0], 1))
    for i in range(x.shape[1]):
        ret = np.append(ret, np.array([np.power(x[:, i], j) for j in range(1, d + 1)]).T, axis = 1)

    return ret
