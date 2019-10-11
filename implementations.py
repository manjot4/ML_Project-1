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
    # TODO: Add comments
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return 1 / y.shape[0] * np.sum(np.power(y - tx.dot(w), 2)), w


def ridge_regression(y, tx, lambda_):
    # TODO: Add comments
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
    N = tx.shape[0]

    o = sigmoid(np.matmul(tx, w))

    return -1.0 / N * np.sum(y * np.log(o) + (1 - y) * np.log(1 - o))


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
    N = tx.shape[0]

    e = sigmoid(np.matmul(tx, w)) - y

    return 1.0 / N * np.matmul(tx.T, e)


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
