''' This script does hyperparameter tuning (lambda and gamma) using cross-validation '''

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


def cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma):
    ''' return the loss for regularised logistic regression for the specific kth-fold '''
    ''' splitting the dataset into k-folds for a particular k-value, 
     kth fold - testing set, (k-1 folds) - training set '''
    k_test = k_indices[k]
    k_train = k_indices[(np.arange(k_indices.shape[0]) != k)]
    k_train = k_train.reshape(-1)
    x_train = tx[k_train]
    x_test = tx[k_test]
    y_train = y[k_train]
    y_test = y[k_test]
	# Regularised Logistic Regression, getting weights
    _, weight = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    # calculating loss for train and test data     
    train_loss = compute_loss_reg_logistic_regression(y_train, x_train, lambda_, weight)
    test_loss = compute_loss_reg_logistic_regression(y_test, x_test, lambda_, weight)

    return train_loss, test_loss


def gamma_lambda_selection_cv(y, tx, k_fold, lambdas, initial_w, max_iters, gammas):
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

    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # list for storing minimum loss values for a particular lambda and gamma
    # and then choosing optimal lambda and gamma value with minimum loss
    min_loss = []
    good_lambda = []
    for gamma in range(len(gammas)):
    	log_reg_tr = [] 
    	log_reg_te = []
    	for lambda_ in range(len(lambdas)):
    		train_loss = []
    		test_loss = []
    		for k in range(k_fold):
    			train, test = cross_validation(y, tx, k_indices, k, lambdas[lambda_], initial_w, max_iters, gammas[gamma])
    			train_loss.append(train)
    			test_loss.append(test)
    		log_reg_tr.append(np.mean(train_loss))
        	log_reg_te.append(np.mean(test_loss))	
        optimal_lambda_ = lambdas[np.argmin(log_reg_te)]
        good_lambda.append(optimal_lambda_)
        min_loss.append(log_reg_te[np.argmin(log_reg_te)])
    optimal_lambda_ = lambdas[np.argmin(min_loss)] 
    optimal_gamma = gammas[np.argmin(min_loss)]

    return optimal_lambda_, optimal_gamma


# This is for graphs, will use the code from exercises - will generate graphs later this week. 
# cross_validation_visualization(lambdas, log_reg_tr, log_reg_te) - plots from ex4. 



def cross_validation_single_paramter(y, tx, k_indices, k, parameter, initial_w, max_iters, function):
    ''' parameter is a hyperparameter which can be either gamma or lambda '''
    ''' return the loss for a particluar model (function) for the specific kth-fold '''
    ''' splitting the dataset into k-folds for a particular k-value, 
     kth fold - testing set, (k-1 folds) - training set '''

    k_test = k_indices[k]
    k_train = k_indices[(np.arange(k_indices.shape[0]) != k)]
    k_train = k_train.reshape(-1)
    x_train = tx[k_train]
    x_test = tx[k_test]
    y_train = y[k_train]
    y_test = y[k_test]

    if str(function) == "least_squares_GD": #optimal gamma
        gamma = parameter
        _, weight = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
        train_loss = compute_loss_least_squares(y_train, x_train, weight)
        test_loss = compute_loss_least_squares(y_test, x_test, weight)

        return train_loss, test_loss

    elif str(function) == "least_squares_SGD": #optimal gamma
        gamma = parameter
        batch_size = 1
        _, weight = least_squares_SGD(y_train, x_train, initial_w, batch_size, max_iters, gamma)
        train_loss = compute_loss_least_squares(y_train, x_train, weight)
        test_loss = compute_loss_least_squares(y_test, x_test, weight)

        return train_loss, test_loss

    elif str(function) == "logistic_regression": #optimal gamma
        gamma = parameter
        _, weight = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
        train_loss = compute_loss_logistic_regression(y_train, x_train, weight)
        test_loss = compute_loss_logistic_regression(y_test, x_test, weight)

        return train_loss, test_loss

    elif str(function) == "ridge_regression": #optimal lambda_
        lambda_ = parameter
        _, weight = ridge_regression(y_train, x_train, lambda_)
        # train_loss = # add loss for ridge regression please and use (y_train, x_train)
        # test_loss = # add loss for ridge regression please and use (y_test, x_test)

        return #train_loss, test_loss


''' just gamma, only used in 3 models i.e. least_squares_GD, least_squares_SGD, logistic_regression'''
def optimal_gamma_selection(y, tx, k_fold, initial_w, max_iters, gammas, function):
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    log_reg_tr = [] 
    log_reg_te = []
    for i in range(len(gammas)):
        train_loss = []
        test_loss = []
        for j in range(k_fold):
            train, test = cross_validation_single_paramter(y, tx, k_indices, j, gammas[i], initial_w, max_iters, function)
            train_loss.append(train)
            test_loss.append(test)
        log_reg_tr.append(np.mean(train_loss))
        log_reg_te.append(np.mean(test_loss))
    optimal_gamma_ = gammas[np.argmin(log_reg_te)]     
    return optimal_gamma_


''' optimal lambda_ - only for Ridge Regression''' 
def optimal_lambda_selection(y, tx, k_fold, initial_w, max_iters, lambdas, function):
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    log_reg_tr = [] 
    log_reg_te = []
    for i in range(len(lambdas)):
        train_loss = []
        test_loss = []
        for j in range(k_fold):
            train, test = cross_validation_single_paramter(y, tx, k_indices, j, lambdas[i], initial_w, max_iters, function)
            train_loss.append(train)
            test_loss.append(test)
        log_reg_tr.append(np.mean(train_loss))
        log_reg_te.append(np.mean(test_loss))
    optimal_lambda_ = lambdas[np.argmin(log_reg_te)]     
    return optimal_lambda_


'''Calculating accuracy'''
def accuracy(y, tx, weight):
    predictions = predict_labels(weight, tx)
    correct_predictions = len(np.intersect1d(y, predictions))
    acc = float(correct_predictions) / len(y)
    return acc

def cross_validation_accuracy(y, tx, k, k_indices, weight):
    '''Calculating accuracy of train and test subset for a particular k-fold'''
    k_test = k_indices[k]
    k_train = k_indices[(np.arange(k_indices.shape[0]) != k)]
    k_train = k_train.reshape(-1)
    x_train = tx[k_train]
    x_test = tx[k_test]
    y_train = y[k_train]
    y_test = y[k_test]
    
    train_accuracy = accuracy(y_train, x_train, weight)
    test_accuracy = accuracy(y_test, x_test, weight)
    
    return train_accuracy, test_accuracy

''' Training accuracy using cross-validation ''' 
def verifying_accuracy_cv(y, tx, k_fold, weight):
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    train_accuracy = []
    test_accuracy = []
    for j in range(k_fold):
        train, test = cross_validation_accuracy(y, tx, j, k_indices, weight)
        train_accuracy.append(train)
        test_accuracy.append(test)
    training_accuracy = (np.mean(train_accuracy))
    testing_accuracy = (np.mean(test_accuracy))
    return training_accuracy, testing_accuracy