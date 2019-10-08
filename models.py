import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        w = w - gamma * gradient

        ws.append(w)
        losses.append(loss)

    return losses, ws


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(y, tx, w)

        w = w - gamma * gradient

        ws.append(w)
        losses.append(loss)

    return losses, ws


def least_squares(y, tx):
    pass


def ridge_regression(y, tx, lambda_):
    pass


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    losses = []
    ws = [initial_w]
    w = initial_w

    for n_iter in range(max_iters):
        loss = compute_loss_logistic_reg(y, tx, w)
        grad = compute_gradient_logistic(y, tx, w)
        w = w - (gamma*grad)
        ws.append(w)
        losses.append(loss)

    return losses, ws



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    losses = []
    ws = [initial_w]
    w = initial_w

    for n_iter in range(max_iters):
        loss, grad = reg_logistic_regression(y, tx, w, lambda_)
        w = w - (gamma*grad)
        ws.append(w)
        losses.append(loss)

    return losses, ws        



def compute_loss(y, tx, w):
    N = tx.shape[0]

    e = y - np.matmul(tx, w)

    return 1.0 / (2 * N) * np.sum(e ** 2)


def compute_gradient(y, tx, w):
    N = tx.shape[0]

    e = y - np.matmul(tx, w)

    return - 1.0 / N * np.matmul(tx.T, e)


def compute_stoch_gradient(y, tx, w):
    N = tx.shape[0]

    e = y - np.matmul(tx, w)

    return - 1.0 / N * np.matmul(tx.T, e)

def reg_logistic_regression(y, tx, w, lambda_):
    N = len(y)
    loss = compute_loss_logistic_reg(y, tx, w) + (lambda_/2)*(np.matmul(w.T, w))
    grad = compute_gradient_logistic(y, tx, w) + lambda_*w
    return loss, grad

# Logistic_Regression_gradient
def compute_gradient_logistic(y, tx, w):
    out = sigmoid_fn(np.matmul(tx, w))
    grad = np.matmul(tx.T, (out-y))
    return grad

# Logistic_Regression_Loss
def compute_loss_logistic_reg(y, tx, w):
    out = sigmoid_fn(np.matmul(tx, w))
    loss = np.matmul((y.T), np.log(out)) + np.matmul(((1-y).T), np.log(1-out))
    return -loss 

# Sigmoid Function
def sigmoid_fn(z):
    den = 1 + np.exp(-z)
    return 1.0/den


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
