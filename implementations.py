import numpy as np
from matplotlib import pyplot as plt


# Whether to enable ploting or not
__should_plot = False


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

    return losses, w


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
        gradient = compute_gradient_reg_logistic_regression(y, tx, lambda_, w)
        loss = compute_loss_reg_logistic_regression(y, tx, lambda_, w)
        w = w - gamma * gradient
        losses.append(loss)

    __plot_loss(losses, "Regularized Logistic Regression using Gradient Descent")

    return losses, w


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
    predictions = sigmoid(tx.dot(w))
    return -(np.log(predictions).T.dot(y) + (1 - np.log(predictions)).T.dot(1 - y))


def compute_loss_reg_logistic_regression(y, tx, lambda_, w):
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


def compute_gradient_reg_logistic_regression(y, tx, lambda_, w):
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


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# CAN GENERATE PREDICTIONS WITHOUT CROSS VALIDATION
# # Cross Validation - STILL NEED TO COMMENT AND GENERALISE
# # we can choose optimal lambda and optimal learning rate through cross validation

# def build_k_indices(y, k_fold, seed):
#     num_row = y.shape[0]
#     interval = int(num_row / k_fold)
#     np.random.seed(seed)
#     indices = np.random.permutation(num_row)
#     k_indices = [indices[k * interval: (k + 1) * interval]
#                  for k in range(k_fold)]
#     return np.array(k_indices)

# def cross_validation(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):
#     k_test = k_indices[k]
#     k_train = k_indices[(np.arange(k_indices.shape[0]) != k)]
#     x_train = x[k_train]
#     x_test = x[k_test]
#     y_train = y[k_train]
#     y_test = y[k_test]

#     # regularised logistic regression(getting weights), one can change these functions in respect to their model 
#     _ , weight = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    
#     # calculating loss for train and test data, one can change these functions in respect to their model      
#     loss_tr = compute_loss_logistic_reg(y_train, x_train, weight, lambda_)
#     loss_te = compute_mse(y_test, x_test, weight, lambda_)
#     return loss_tr, loss_te

# CHOOSING BEST LAMBDA - will add alpha in this code later
# def best_lambda_selection(y, tx, k_fold, lambdas):
#     seed = 1
#     k_fold = 10 #will be given as an argument
#     lambdas = np.logspace(-4, 0, 30) #will be given as an argument
#     # split data in k fold
#     k_indices = build_k_indices(y, k_fold, seed)
#     # define lists to store the loss of training data and test data
#     rmse_tr = [] #change loss names
#     rmse_te = []
#     for i in range(len(lambdas)):
#         tr_loss = []
#         te_loss = []
#         for i in range(k_fold):
#             tr, te = cross_validation(y, x, k_indices, i, lambdas[i], initial_w, max_iters, gamma)
#             tr_loss.append(tr)
#             te_loss.append(te)
#         rmse_tr.append(tr_loss.mean())
#         rmse_te.append(te_loss.mean())
#     lambda_optimal = lambdas[np.argmin(rmse_te)] 
    
#     return lambda_optimal
#     # cross_validation_visualization(lambdas, rmse_tr, rmse_te)


## Build_Polynomial - Basis Function.....
# def build_poly(tx, degree):
#     """polynomial basis functions for input data x, for j=0 up to j=degree."""
#     polynomial = np.ones(tx.shape[0], 1))
#     for i in range(1, degree+1):
# #         x_degree = np.power(tx,i)
#         polynomial = np.c_[polynomial, np.power(tx, i)]
# #     polynomial = np.sum(polynomial, axis=1)
#     return polynomial


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
