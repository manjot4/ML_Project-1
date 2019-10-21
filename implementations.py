'''This script contains implementations (computation of loss, gradient, polynomial building) of all the models used in this project'''

import numpy as np
from matplotlib import pyplot as plt


# Whether to enable ploting or not
__should_plot = False


# Main functions

def __plot_loss(losses, title):
    """ Utility function that plots the train losses.

    Parameters
    ----------
    losses: array
        The losses
    title: string
        The title of the plot
    """
    if __should_plot:
        plt.plot(losses)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()


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

    __plot_loss(losses, "Least Squares using Gradient Descent")

    return losses[-1], w


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

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_gradient_least_squares(minibatch_y, minibatch_tx, w)
        loss = compute_loss_least_squares(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)

    __plot_loss(losses, "Least Squares using Stochastic Gradient Descent")

    return losses[-1], w


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
        The last loss and optimal weights
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return 1 / y.shape[0] * np.sum(np.power(y - tx.dot(w), 2)), w


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
        The last loss and optimal weights
    """
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.identity(tx.shape[1]), tx.T.dot(y))
    return 1 / y.shape[0] * np.sum(np.power(y - tx.dot(w), 2)), w


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

    __plot_loss(losses, "Logistic Regression using Gradient Descent")

    return losses[-1], w


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

    __plot_loss(losses, "Regularized (L2) Logistic Regression using Gradient Descent")

    return losses[-1], w


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

    __plot_loss(losses, "Regularized (L1) Logistic Regression using Gradient Descent")

    return losses[-1], w




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
        The last loss and learned weights
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
        The last loss and learned weights
    """
    eps = 1e-5
    predictions = sigmoid(tx.dot(w))
    #print(w)
    assert predictions.min() > -eps # make sure numbers are close to 0
    loss = y.T.dot(np.log(predictions + eps)) + (1 - y).T.dot(np.log(1 - predictions + eps)) # need to justify this
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
        The last loss and learned weights
    """
    return compute_loss_logistic_regression(y, tx, w) + (lambda_ / 2) * np.matmul(w.T, w)


def compute_loss_reg_logistic_regression_L1(y, tx, w, lambda_):
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
        The last loss and learned weights
    """
    return compute_loss_logistic_regression(y, tx, w) + lambda_ * np.sum(np.absolute(w))


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

    Returns
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
    return compute_gradient_logistic_regression(y, tx, w) - lambda_ * w


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


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    """
    Takes a dataset of N examples and D features (NxD array) and creates polynomials
    of the features, up to degree 'degree'.
    
    The first column is always 1 as it represents all polynomials with degree 0 for all D features.
    Then the first 'degree' columns represent the polynomials of the first feature, the following 'degree'
    columns represent the polynomials for the second feature and so on.
    
    Thus, the returned dataset has size Nx(1 + D * 'degree').
    """
    
    # handle the case when D = 1
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    ret = np.ones((x.shape[0], 1))
    for i in range(x.shape[1]):
        ret = np.append(ret, np.array([np.power(x[:, i], j) for j in range(1, degree + 1)]).T, axis = 1)
    return ret